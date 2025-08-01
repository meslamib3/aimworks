# ------------------------------ hydrogen_ontology_generator.py ------------------------------
"""
Hydrogen Technology Ontology Generator (EMMO-based)

Full, functional Streamlit app implementing:
  - Multi-agent pipeline with hallucination mitigation (critic review, deterministic formatting)
  - EMMO alignment + reuse of core ontology classes (e.g., EMMO, QUDT, ChEBI, PROV integration)
  - SHACL-based symbolic validation of relations (optional – skips if `pyshacl` not installed)
  - Interactive user confirmation for features and relations (synonym merging and relation editing)
  - Detailed logging via Streamlit UI for traceability and reproducibility
"""
import streamlit as st
import os, re, json, datetime, textwrap
from typing import List, Dict, Optional

# Use Streamlit secrets for deployment (e.g., Streamlit Cloud)
if "OPENAI_API_KEY" in st.secrets:
    os.environ.setdefault("OPENAI_API_KEY", st.secrets["OPENAI_API_KEY"])

# -- LLM & multi-agent orchestration (CrewAI) --
try:
    import openai
    from crewai import Agent, Task, Crew, Process
    from crewai.llm import LLM
except ImportError as e:
    st.error(f"Missing library **{e.name}**. Please install it to use this app.\n\n"
             "```bash\npip install crewai openai\n```")
    st.stop()

# -- Semantic Web stack (RDF, Graph) --
try:
    from rdflib import Graph, Namespace, RDF, RDFS, OWL, XSD, Literal, URIRef, BNode
    import networkx as nx
    from graphviz import Digraph
    import requests
except ImportError as e:
    st.error(f"Missing dependency **{e.name}**. Please install it to use this app.\n\n"
             "```bash\npip install rdflib networkx graphviz requests\n```")
    st.stop()

# -- SHACL validation (optional) --
try:
    from pyshacl import validate as pyshacl_validate
    SHACL_AVAILABLE = True
except ImportError:
    SHACL_AVAILABLE = False  # not a hard stop; we'll skip validation if unavailable

# -- Streamlit page config --
st.set_page_config(page_title="Hydrogen Ontology Generator", page_icon="??", layout="centered")
st.title("?? Hydrogen Technology Ontology Generator (EMMO-based)")

st.write(
    "**Description:** Provide a hydrogen technology research question (and optionally a list of features) to automatically generate an extended ontology. "
    "Multiple GPT-4 agents (feature scientist, terminologist, domain ontologist, etc.) collaborate to identify relevant concepts, merge synonyms, align with external standards (EMMO, QUDT, ChEBI, PROV), and assemble an OWL ontology. "
    "A review step allows you to confirm or edit proposed relationships before finalizing. "
    "The final ontology can be downloaded in Turtle format, and a graph visualization is provided for inspection."
)

# -- UI Inputs --
question: str = st.text_input("**Research Question:**",
    help="E.g. 'How does operating temperature affect hydrogen permeability of a Nafion membrane in PEM fuel cells?'")
api_key_default = os.getenv("OPENAI_API_KEY", "")
api_key: str = st.text_input("**OpenAI API Key** (required for GPT-4):", type="password", value=api_key_default)

# -- Initialize session state --
if "features_text" not in st.session_state:
    st.session_state["features_text"] = ""
if "ontology_generated" not in st.session_state:
    st.session_state["ontology_generated"] = False
if "relation_review_done" not in st.session_state:
    st.session_state["relation_review_done"] = False
if "structure" not in st.session_state:
    st.session_state["structure"] = {}
if "synonyms_map" not in st.session_state:
    st.session_state["synonyms_map"] = {}
if "last_ont_uri" not in st.session_state:
    st.session_state["last_ont_uri"] = None

# -- Utility functions --
CAMEL_RE = re.compile(r'(?<=[a-z])(?=[A-Z])')

def to_valid_uri(label: str) -> str:
    """Convert an arbitrary label to a CamelCase string suitable for a URI fragment."""
    parts = re.split(r"[\W_]+", label)
    return "".join(word.capitalize() for word in parts if word) or label

def md_box(msg: str, state: str = "info"):
    """Utility to display a Markdown-styled message in a colored box (info, warning, error)."""
    getattr(st, state)(textwrap.dedent(msg).strip())

# Feature suggestion (Hydrogen-Feature Scientist agent)
def suggest_features(question_text: str) -> str:
    if not question_text.strip():
        return ""
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    llm = LLM(model="openai/gpt-4", temperature=0.2, max_tokens=512)
    scientist = Agent(
        role="Hydrogen-Feature Scientist",
        goal="Identify key materials, properties, parameters, and processes in hydrogen tech relevant to the question.",
        backstory=("You are a senior researcher in hydrogen technology. You know the important experimental factors, materials, and parameters in this domain. "
                   "Your goal is to list the most relevant features (CamelCase terms) related to the research question."),
        llm=llm
    )
    prompt = (
        "List **5-15** relevant features (materials, properties, parameters, processes, etc.) related to the following research question. "
        "Provide them as a single comma-separated line (CamelCase for multi-word terms, no explanations):\n\n"
        f"Research question: \"{question_text}\""
    )
    task = Task(description=prompt, expected_output="Comma-separated list of features", agent=scientist)
    Crew(agents=[scientist], tasks=[task], process=Process.sequential).kickoff()
    output = str(task.output).strip().strip("`")
    # Remove any surrounding quotes or markdown artifacts
    output = output.strip(", ")
    return output

# Synonym merging (Lexical Synonym Expert agent)
def find_synonyms(feature_list: List[str]) -> Dict[str, List[str]]:
    if not feature_list:
        return {}
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    llm = LLM(model="openai/gpt-4", temperature=0.0, max_tokens=500)
    terminologist = Agent(
        role="Lexical Synonym Expert",
        goal="Merge equivalent terms or abbreviations among the features",
        backstory=("You are an ontology terminologist with expertise in hydrogen domain nomenclature, identifying synonyms and abbreviations."),
        llm=llm
    )
    feature_str = "; ".join(feature_list)
    syn_prompt = (
        "The feature list may contain synonyms or abbreviations referring to the same concept. "
        "Group equivalent terms and choose one canonical name per group. "
        "Output JSON mapping each **canonical term** to a list of its synonyms/alternates from the list (use exact original spelling for synonyms). "
        "If a term has no synonyms in the list, include it with an empty list.\n\n"
        f"Features: {feature_str}"
    )
    task = Task(description=syn_prompt, expected_output='JSON { "CanonicalTerm": ["Synonym1", "Synonym2", ...], ... }', agent=terminologist)
    Crew(agents=[terminologist], tasks=[task], process=Process.sequential).kickoff()
    raw = str(task.output).strip().strip("`")
    try:
        syn_map = json.loads(raw)
    except json.JSONDecodeError:
        # Try a common fix: replace single quotes with double quotes
        try:
            syn_map = json.loads(raw.replace("'", "\""))
        except Exception:
            syn_map = {}
    cleaned_map: Dict[str, List[str]] = {}
    if isinstance(syn_map, dict):
        for canon, alts in syn_map.items():
            if isinstance(canon, str):
                canon_clean = canon.strip()
                if not canon_clean:
                    continue
                if isinstance(alts, list):
                    cleaned_map[canon_clean] = [str(a).strip() for a in alts if isinstance(a, str) and a.strip()]
                else:
                    cleaned_map[canon_clean] = []
    return cleaned_map

# Ontology structuring (Domain Expert + Ontology Engineer agents, plus critic)
def build_ontology_structure(question_text: str, feature_list: List[str]) -> Optional[Dict]:
    if not feature_list:
        return None
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    # Domain expert agent (classifies features and suggests relations)
    domain_llm = LLM(model="openai/gpt-4", temperature=0.3, max_tokens=1200)
    domain_agent = Agent(
        role="Hydrogen Domain Expert",
        goal="Categorize features and propose meaningful relations",
        backstory=("You are an expert in hydrogen technology ontology. You will categorize each feature into an ontology class type and suggest only plausible hasProperty or influences relationships."),
        llm=domain_llm
    )
    # Formatting agent (ensures structured JSON output)
    format_llm = LLM(model="openai/gpt-4", temperature=0.0, max_tokens=800)
    engineer_agent = Agent(
        role="Ontology Engineer",
        goal="Format the ontology classes and relations into JSON",
        backstory=("You convert the domain expert's analysis into a structured JSON ontology representation."),
        llm=format_llm
    )
    # Prompt for domain agent
    example = (
        "Feature list example – Membrane, HydrogenPermeability, TestingTemperature\n"
        "Correct JSON format –\n"
        "{\n"
        '  "materials": ["Membrane"],\n'
        '  "properties": ["HydrogenPermeability"],\n'
        '  "parameters": ["TestingTemperature"],\n'
        '  "manufacturings": [], "measurements": [], "simulations": [], "metadata": [],\n'
        '  "relationships": [\n'
        '    {"subject": "Membrane", "predicate": "hasProperty", "object": "HydrogenPermeability"},\n'
        '    {"subject": "TestingTemperature", "predicate": "influences", "object": "HydrogenPermeability"}\n'
        '  ]\n'
        '}'
    )
    analysis_prompt = (
        f"Research Question: {question_text or 'N/A'}\n"
        f"Features: {', '.join(feature_list)}\n\n"
        "Classify each feature into one of the categories: Material, Property, Parameter, Manufacturing (process), Measurement (process), Simulation (process), or Metadata (contextual). "
        "Then suggest plausible relationships among them using only:\n"
        "- **hasProperty**: (Material -> Property) to indicate a material has a property.\n"
        "- **influences**: (Process/Parameter -> Property) to indicate a condition or process affects a property.\n"
        "Only include relationships that make scientific sense. If unsure, omit the relation. Do not introduce other relation types.\n\n"
        "Finally, list the categories and relationships clearly.\n"
        f"{example}"
    )
    format_prompt = (
        "Now format the ontology content as JSON with keys: "
        "'materials', 'properties', 'parameters', 'manufacturings', 'measurements', 'simulations', 'metadata', and 'relationships'. "
        "Each key should map to a list of feature names (unique, no duplicates). "
        "'relationships' should be a list of objects each with 'subject', 'predicate', 'object'. "
        "Use exactly 'hasProperty' or 'influences' for predicates. "
        "Only output the JSON object, no extra text."
    )
    t1 = Task(description=analysis_prompt, expected_output="Categorization and relation analysis (text)", agent=domain_agent)
    t2 = Task(description=format_prompt, expected_output="JSON ontology structure", agent=engineer_agent, context=[t1])
    Crew(agents=[domain_agent, engineer_agent], tasks=[t1, t2], process=Process.sequential).kickoff()
    raw_output = str(t2.output).strip().strip("`")
    # Extract JSON substring if any extraneous text
    start_idx = raw_output.find("{")
    end_idx = raw_output.rfind("}")
    if start_idx != -1 and end_idx != -1:
        raw_json_str = raw_output[start_idx: end_idx+1]
    else:
        raw_json_str = raw_output
    try:
        structure = json.loads(raw_json_str)
    except json.JSONDecodeError:
        # Attempt to fix common JSON issues (single quotes, trailing commas)
        fix_str = raw_json_str.replace("'", "\"")
        fix_str = re.sub(r",\s*}", "}", fix_str)  # remove trailing comma
        fix_str = re.sub(r",\s*\]", "]", fix_str)
        try:
            structure = json.loads(fix_str)
        except Exception as e:
            st.error("Ontology structuring failed: the AI output was not valid JSON.")
            st.text(raw_output)  # display raw output for debugging
            return None
    # Ensure all original features are categorized; if not, add them as Metadata
    input_features_set = set(feature_list)
    categorized_set = set()
    for cat_key in ["materials", "properties", "parameters", "manufacturings", "measurements", "simulations", "metadata"]:
        for item in structure.get(cat_key, []):
            categorized_set.add(item)
    missing_feats = input_features_set - categorized_set
    if missing_feats:
        structure.setdefault("metadata", [])
        for item in missing_feats:
            if item not in structure["metadata"]:
                structure["metadata"].append(item)
        st.warning(f"Uncategorized features were added as Metadata: {', '.join(missing_feats)}")
    # Deduplicate entries in category lists
    for cat_key in ["materials", "properties", "parameters", "manufacturings", "measurements", "simulations", "metadata"]:
        if cat_key in structure and isinstance(structure[cat_key], list):
            seen = set()
            unique_list = []
            for val in structure[cat_key]:
                if val not in seen:
                    unique_list.append(val)
                    seen.add(val)
            structure[cat_key] = unique_list
    return structure

# Critic agent to review the drafted ontology structure
def critique_structure(structure: Dict) -> str:
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    critic_llm = LLM(model="openai/gpt-4", temperature=0.0, max_tokens=400)
    critic = Agent(
        role="Ontology Critic",
        goal="Evaluate ontology relations for plausibility and completeness",
        backstory=("You are a critical reviewer identifying any implausible or missing relations in a hydrogen technology ontology draft."),
        llm=critic_llm
    )
    critique_prompt = (
        "Review the JSON ontology draft below. List any relation that seems questionable or any obvious missing relation (one issue per line). "
        "If everything looks plausible, return 'OK'.\n\n"
        f"{json.dumps(structure, indent=2)}"
    )
    task = Task(description=critique_prompt, expected_output="Critique or OK", agent=critic)
    Crew(agents=[critic], tasks=[task], process=Process.sequential).kickoff()
    return str(task.output).strip()

# Ontology graph assembly
def assemble_ontology_graph(structure: Dict, base_graph: Optional[Graph] = None) -> (Graph, Namespace):
    G = Graph()
    # Incorporate base ontology (EMMO/MatGraph) if provided
    if base_graph:
        G += base_graph
    # Determine base namespace for our ontology extension
    base_ns = None
    if base_graph:
        for subj in base_graph.subjects(RDF.type, OWL.Ontology):
            base_ns = Namespace(str(subj) + "_ext#")
            break
    if base_ns is None:
        base_ns = Namespace("http://example.org/hydrogen-ontology#")
    # Bind common prefixes
    G.bind("mat", base_ns)
    G.bind("owl", OWL); G.bind("rdf", RDF); G.bind("rdfs", RDFS); G.bind("xsd", XSD)
    G.bind("skos", Namespace("http://www.w3.org/2004/02/skos/core#"))
    G.bind("prov", Namespace("http://www.w3.org/ns/prov#"))
    G.bind("qudt", Namespace("http://qudt.org/schema/qudt/"))
    G.bind("quantitykind", Namespace("http://qudt.org/vocab/quantitykind/"))
    G.bind("chebi", Namespace("http://purl.obolibrary.org/obo/CHEBI_"))
    # Core ontology classes (if not already present via base_graph)
    core_classes = {
        "Material": base_ns.Material,
        "Property": base_ns.Property,
        "Parameter": base_ns.Parameter,
        "Process": base_ns.Process,
        "Metadata": base_ns.Metadata
    }
    for name, uri in core_classes.items():
        if (uri, RDF.type, OWL.Class) not in G:
            G.add((uri, RDF.type, OWL.Class))
    # Define subclass classes for specific process types
    if structure.get("manufacturings"):
        subclass = base_ns.Manufacturing
        G.add((subclass, RDF.type, OWL.Class))
        G.add((subclass, RDFS.subClassOf, base_ns.Process))
    if structure.get("measurements"):
        subclass = base_ns.Measurement
        G.add((subclass, RDF.type, OWL.Class))
        G.add((subclass, RDFS.subClassOf, base_ns.Process))
    if structure.get("simulations"):
        subclass = base_ns.Simulation
        G.add((subclass, RDF.type, OWL.Class))
        G.add((subclass, RDFS.subClassOf, base_ns.Process))
    # Add feature classes and subclass relationships
    category_parent = {
        "materials": base_ns.Material,
        "properties": base_ns.Property,
        "parameters": base_ns.Parameter,
        "manufacturings": base_ns.Manufacturing if structure.get("manufacturings") else base_ns.Process,
        "measurements": base_ns.Measurement if structure.get("measurements") else base_ns.Process,
        "simulations": base_ns.Simulation if structure.get("simulations") else base_ns.Process,
        "metadata": None
    }
    for cat_key, parent_class in category_parent.items():
        for item in structure.get(cat_key, []):
            cls_uri = base_ns[to_valid_uri(item)]
            if (cls_uri, RDF.type, OWL.Class) not in G:
                G.add((cls_uri, RDF.type, OWL.Class))
            if parent_class and cat_key != "metadata":
                G.add((cls_uri, RDFS.subClassOf, parent_class))
    # Define core object properties if not present
    hasP = base_ns.hasProperty
    infl = base_ns.influences
    if (hasP, RDF.type, None) not in G:
        G.add((hasP, RDF.type, OWL.ObjectProperty))
        G.add((hasP, RDFS.domain, base_ns.Material))
        G.add((hasP, RDFS.range, base_ns.Property))
        G.add((hasP, RDFS.label, Literal("has property", lang="en")))
    if (infl, RDF.type, None) not in G:
        G.add((infl, RDF.type, OWL.ObjectProperty))
        G.add((infl, RDFS.domain, base_ns.Process))
        G.add((infl, RDFS.range, base_ns.Property))
        G.add((infl, RDFS.label, Literal("influences", lang="en")))
    # Extended data properties and PROV alignment
    val_prop = base_ns.hasValue
    unit_prop = base_ns.hasUnit
    exp_class = base_ns.Experiment
    if (val_prop, RDF.type, None) not in G:
        G.add((val_prop, RDF.type, OWL.DatatypeProperty))
        G.add((val_prop, RDFS.domain, base_ns.Property))
        G.add((val_prop, RDFS.range, XSD.double))
        G.add((val_prop, RDFS.label, Literal("has value", lang="en")))
    if (unit_prop, RDF.type, None) not in G:
        G.add((unit_prop, RDF.type, OWL.ObjectProperty))
        G.add((unit_prop, RDFS.domain, base_ns.Property))
        # range as QUDT Unit class
        G.add((unit_prop, RDFS.range, URIRef("http://qudt.org/schema/qudt/Unit")))
        G.add((unit_prop, RDFS.label, Literal("has unit", lang="en")))
    if (exp_class, RDF.type, None) not in G:
        G.add((exp_class, RDF.type, OWL.Class))
        G.add((exp_class, RDFS.subClassOf, URIRef("http://www.w3.org/ns/prov#Activity")))
        G.add((exp_class, RDFS.label, Literal("Experiment", lang="en")))
    # Add relationships as class-to-class links (OWL object property assertions at class level)
    seen_relations = set()
    for rel in structure.get("relationships", []):
        subj = rel.get("subject"); pred = rel.get("predicate"); obj = rel.get("object")
        if not subj or not pred or not obj:
            continue
        subj_uri = base_ns[to_valid_uri(subj)]
        obj_uri  = base_ns[to_valid_uri(obj)]
        if pred == "hasProperty":
            pred_uri = base_ns.hasProperty
        elif pred == "influences":
            pred_uri = base_ns.influences
        else:
            pred_uri = None
        if pred_uri is None or subj_uri == obj_uri:
            continue
        triple = (subj_uri, pred_uri, obj_uri)
        if triple in seen_relations:
            continue
        seen_relations.add(triple)
        G.add(triple)
    return G, base_ns

# External ontology alignment (Alignment Specialist agent)
def map_to_external(class_list: List[tuple]) -> Dict[str, Dict[str, str]]:
    """Map given classes (name, category) to external ontology concepts."""
    mapping: Dict[str, Dict[str, str]] = {}
    if not class_list:
        return mapping
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    llm = LLM(model="openai/gpt-4", temperature=0.0, max_tokens=600)
    aligner = Agent(
        role="Ontology Alignment Specialist",
        goal="Find equivalent concepts in standard ontologies for given terms",
        backstory=("You are an expert in ontologies (QUDT for quantities, ChEBI for chemicals, EMMO for materials, etc.). Map each class to an external ontology if possible."),
        llm=llm
    )
    classes_str = "\n".join([f"- {name} ({cat})" for name, cat in class_list])
    align_prompt = (
        "For each class (with its category) below, find an equivalent concept in a well-known ontology or vocabulary (if it exists). "
        "Use ontologies such as: ChEBI for chemical substances/materials, QUDT for physical quantities or units, QuantityKind for quantity types, EMMO for generic concepts, or others if relevant. "
        "Output a JSON mapping of the class name to an object with 'ontology' and 'uri'. If no match, you may omit that class or return an empty object for it.\n"
        "Classes:\n" + classes_str
    )
    task = Task(description=align_prompt, expected_output='JSON { "ClassName": {"ontology": "...", "uri": "..."}, ... }', agent=aligner)
    Crew(agents=[aligner], tasks=[task], process=Process.sequential).kickoff()
    raw = str(task.output).strip().strip("`")
    try:
        mapping = json.loads(raw)
    except json.JSONDecodeError:
        try:
            mapping = json.loads(raw.replace("'", "\""))
        except Exception:
            mapping = {}
    # Clean mapping to ensure proper format
    final_map: Dict[str, Dict[str, str]] = {}
    if isinstance(mapping, dict):
        for cls, info in mapping.items():
            if not isinstance(cls, str):
                continue
            if isinstance(info, dict):
                ont = str(info.get("ontology", "")).strip()
                uri = str(info.get("uri", "")).strip()
                if uri:
                    final_map[cls.strip()] = {"ontology": ont, "uri": uri}
    return final_map

def add_external_mappings(G: Graph, mapping: Dict[str, Dict[str, str]], base_ns: Namespace):
    """Add owl:equivalentClass links for mapped external concepts to our ontology graph."""
    for cls_name, info in mapping.items():
        uri_str = info.get("uri")
        if not uri_str:
            continue
        cls_uri = base_ns[to_valid_uri(cls_name)]
        if (cls_uri, RDF.type, OWL.Class) not in G:
            continue  # only map if class is present
        ext_uri = URIRef(uri_str)
        # Bind prefix for external ontology if known
        ont_name = str(info.get("ontology", "")).lower()
        if ont_name.startswith("chebi") or "chebi" in uri_str.lower():
            G.bind("chebi", "http://purl.obolibrary.org/obo/CHEBI_")
            # Convert CHEBI:IDs to full URI if needed
            if uri_str.startswith("CHEBI:"):
                chebi_id = uri_str.split(":")[1]
                ext_uri = URIRef(f"http://purl.obolibrary.org/obo/CHEBI_{chebi_id}")
        elif ont_name.startswith("qudt") or "quantitykind" in uri_str.lower():
            G.bind("quantitykind", "http://qudt.org/vocab/quantitykind/")
        elif ont_name.startswith("emmo"):
            G.bind("emmo", "http://emmo.info/emmo#")
        # Add mapping triple and annotation
        G.add((cls_uri, OWL.equivalentClass, ext_uri))
        if info.get("ontology"):
            G.add((cls_uri, RDFS.comment, Literal(f"Equivalent to {info['ontology']} term {info['uri']}", lang="en")))

def add_labels_and_definitions(G: Graph, synonyms_map: Dict[str, List[str]], structure: Dict, base_ns: Namespace):
    """Add human-readable labels (skos:prefLabel, skos:altLabel) and brief definitions (rdfs:comment) for each class."""
    SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    def_llm = LLM(model="openai/gpt-4", temperature=0.3, max_tokens=100)
    def_agent = Agent(
        role="Ontology Documentation Specialist",
        goal="Provide a concise definition for ontology classes",
        backstory=("You are a scientist writing brief definitions for hydrogen technology terms in an ontology."),
        llm=def_llm
    )
    classes = []  # list of (URI, name, category)
    for cat_key in ["materials", "properties", "parameters", "manufacturings", "measurements", "simulations", "metadata"]:
        for name in structure.get(cat_key, []):
            cat = cat_key[:-1] if cat_key.endswith('s') else cat_key
            if cat == "manufacturing": cat = "manufacturing process"
            if cat == "measurement": cat = "measurement process"
            if cat == "simulation": cat = "simulation process"
            uri = base_ns[to_valid_uri(name)]
            if (uri, RDF.type, OWL.Class) in G:
                classes.append((uri, name, cat))
    for uri, name, category in classes:
        # Preferred label (split CamelCase into words if needed)
        label = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', name)
        G.add((uri, SKOS.prefLabel, Literal(label, lang="en")))
        # Alternate labels for synonyms
        if name in synonyms_map:
            for alt in synonyms_map[name]:
                alt_label = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', alt)
                if alt_label.lower() != label.lower():  # avoid duplicating prefLabel
                    G.add((uri, SKOS.altLabel, Literal(alt_label, lang="en")))
        # Definition (if not already present as RDFS.comment)
        if (uri, RDFS.comment, None) not in G:
            prompt = f"Provide a one-sentence definition for '{name}' as a {category} in the context of hydrogen technology."
            task = Task(description=prompt, expected_output="Brief definition sentence.", agent=def_agent)
            Crew(agents=[def_agent], tasks=[task], process=Process.sequential).kickoff()
            definition = str(task.output).strip().strip("`")
            if definition:
                # Ensure it ends with a period
                if definition[-1] not in ".!?":
                    definition += "."
                G.add((uri, RDFS.comment, Literal(definition, lang="en")))

def apply_versioning(G: Graph, base_graph: Optional[Graph]):
    """Add ontology metadata like versioning info and imports, and track in session state."""
    ont_uri = URIRef("http://example.org/hydrogen-ontology")
    if base_graph:
        # If extending an existing ontology, import it and name this as an extension
        for subj in base_graph.subjects(RDF.type, OWL.Ontology):
            ont_uri = URIRef(str(subj) + "_extended")
            G.add((ont_uri, OWL.imports, subj))
            break
    G.add((ont_uri, RDF.type, OWL.Ontology))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    G.add((ont_uri, OWL.versionInfo, Literal(f"Generated on {timestamp}", datatype=XSD.string)))
    # Link to previous version if exists
    if st.session_state.get("last_ont_uri"):
        prev_uri = URIRef(st.session_state["last_ont_uri"])
        G.add((ont_uri, OWL.priorVersion, prev_uri))
    # Update last_ont_uri in session state
    st.session_state["last_ont_uri"] = str(ont_uri)

def run_shacl_validation(G: Graph, base_ns: Namespace) -> str:
    """Run SHACL shape validation on influences/hasProperty usage. Returns empty string if all good, or report text if errors (or 'SKIPPED')."""
    if not SHACL_AVAILABLE:
        return "SKIPPED (pyshacl not installed)"
    
    sh = Graph()
    SH = Namespace("http://www.w3.org/ns/shacl#")
    sh.bind("sh", SH)
    sh.bind("mat", base_ns)

    # Shape: subject of influences must be a Process
    shape1 = BNode()
    sh.add((shape1, RDF.type, SH.NodeShape))
    sh.add((shape1, SH.targetSubjectsOf, base_ns.influences))
    sh.add((shape1, SH["class"], base_ns.Process))

    # Shape: object of influences must be a Property
    shape2 = BNode()
    sh.add((shape2, RDF.type, SH.NodeShape))
    sh.add((shape2, SH.targetObjectsOf, base_ns.influences))
    sh.add((shape2, SH["class"], base_ns.Property))

    # Shape: subject of hasProperty must be a Material
    shape3 = BNode()
    sh.add((shape3, RDF.type, SH.NodeShape))
    sh.add((shape3, SH.targetSubjectsOf, base_ns.hasProperty))
    sh.add((shape3, SH["class"], base_ns.Material))

    # Shape: object of hasProperty must be a Property
    shape4 = BNode()
    sh.add((shape4, RDF.type, SH.NodeShape))
    sh.add((shape4, SH.targetObjectsOf, base_ns.hasProperty))
    sh.add((shape4, SH["class"], base_ns.Property))

    conforms, report_graph, report_text = pyshacl_validate(
        data_graph=G,
        shacl_graph=sh,
        inference='rdfs'
    )

    if conforms:
        return ""
    else:
        return report_text.decode("utf-8") if isinstance(report_text, bytes) else str(report_text)

# -- Feature Suggestion Action --
if st.button("?? Suggest Features"):
    if not api_key:
        st.error("Please provide your OpenAI API key.")
    else:
        os.environ["OPENAI_API_KEY"] = api_key.strip()
        with st.spinner("GPT-4 is analyzing the question to suggest features..."):
            suggested = suggest_features(question)
        if suggested:
            st.session_state["features_text"] = suggested
            st.success("Suggested features have been added. You can review or edit them below.")
        else:
            st.warning("No features were suggested. The question might be too broad or unclear. You can manually enter features below.")

# -- Feature list input (editable) --
features_text = st.text_area("**Feature List** (comma-separated):", 
                             value=st.session_state.get("features_text", ""), 
                             height=100, 
                             help="List of features to include in the ontology. You can edit or add to this list before generating the ontology.")
features = [f.strip() for f in features_text.split(",") if f.strip()]
st.session_state["features_text"] = features_text  # update session state with any manual edits

if features and not question.strip():
    st.info("No specific question provided. A general ontology will be generated from the features list.")

# -- Generate Ontology Action --
if st.button("?? Generate Ontology"):
    if not api_key:
        st.error("Please provide an OpenAI API key to proceed.")
        st.stop()
    if not features:
        st.error("Feature list is empty. Please enter at least one feature.")
        st.stop()
    # Set API key for subsequent agent calls
    os.environ["OPENAI_API_KEY"] = api_key.strip()
    # Step 1: Synonym merging
    syn_map = find_synonyms(features)
    if syn_map:
        # Prepare merged feature list (canonical terms)
        merged_features = list(syn_map.keys())
        features = merged_features
        # Display synonym merges
        merge_info_strs = []
        for canon, alts in syn_map.items():
            if alts:
                merge_info_strs.append(f"**{canon}** (merged: {', '.join(alts)})")
            else:
                merge_info_strs.append(f"**{canon}**")
        st.write("**Synonym Merging:** The following terms were identified as duplicates or aliases and will be unified:")
        st.write("; ".join(merge_info_strs))
        st.info("If any merge is incorrect, please modify the feature list above and regenerate.")
    else:
        syn_map = {}
    # Step 2: Build ontology structure with GPT-4
    with st.spinner("Classifying features and proposing relations..."):
        structure = build_ontology_structure(question, features)
    if not structure:
        st.stop()  # Error already shown in build_ontology_structure if any
    # Step 3: Critique the proposed structure
    critic_feedback = critique_structure(structure)
    if critic_feedback and critic_feedback.lower() != "ok":
        md_box("### ?? Critic Feedback", "warning")
        st.write(critic_feedback)
        # By default, offer to remove any flagged relations
        if st.checkbox("Apply critic suggestions (remove flagged relations)", value=True):
            flagged = []
            for rel in structure.get("relationships", []):
                rel_str = f"{rel.get('subject')} {rel.get('predicate')} {rel.get('object')}"
                for line in critic_feedback.splitlines():
                    if rel_str in line:
                        flagged.append(rel)
                        break
            if flagged:
                structure["relationships"] = [r for r in structure["relationships"] if r not in flagged]
                md_box(f"Removed {len(flagged)} relation(s) flagged by critic.")
    else:
        critic_feedback = "OK"
    # Store intermediate results in session state for confirmation step
    st.session_state["structure"] = structure
    st.session_state["synonyms_map"] = syn_map
    st.session_state["ontology_generated"] = True
    # Inform user to proceed to relation confirmation
    st.success("Ontology structure generated! Review the proposed relationships below before finalizing.")

# -- Relation confirmation UI (if ontology structure is generated and not yet confirmed) --
if st.session_state.get("ontology_generated") and not st.session_state.get("relation_review_done"):
    st.subheader("?? Confirm or Edit Proposed Relationships")
    rel_lines = []
    for rel in st.session_state["structure"].get("relationships", []):
        subj = rel.get("subject"); pred = rel.get("predicate"); obj = rel.get("object")
        if subj and pred and obj:
            rel_lines.append(f"{subj} {pred} {obj}")
    relations_text = st.text_area("One relation per line (format: Subject predicate Object):", 
                                  value="\n".join(rel_lines), height=100)
    st.write("*(You can remove or modify any relationships. Use only 'hasProperty' or 'influences' as predicates.)*")
    if st.button("? Confirm Relations"):
        final_rels = []
        for line in relations_text.splitlines():
            parts = line.strip().split()
            if len(parts) == 3 and parts[1] in {"hasProperty", "influences"}:
                subj, pred, obj = parts
                final_rels.append({"subject": subj, "predicate": pred, "object": obj})
        st.session_state["structure"]["relationships"] = final_rels
        # If any new classes appear in relations that were not in the category lists, add them as Metadata
        all_classes = {item for cat in ["materials","properties","parameters","manufacturings","measurements","simulations","metadata"] for item in st.session_state["structure"].get(cat, [])}
        newly_introduced = set()
        for rel in final_rels:
            if rel["subject"] not in all_classes:
                newly_introduced.add(rel["subject"])
            if rel["object"] not in all_classes:
                newly_introduced.add(rel["object"])
        if newly_introduced:
            st.warning(f"The following terms from relations were not in the feature list and will be added as 'Metadata': {', '.join(newly_introduced)}")
            st.session_state["structure"].setdefault("metadata", [])
            for item in newly_introduced:
                if item not in st.session_state["structure"]["metadata"]:
                    st.session_state["structure"]["metadata"].append(item)
        # Mark confirmation done
        st.session_state["relation_review_done"] = True
        st.success("Relations confirmed. Ontology is ready to build.")

# -- Build final ontology and outputs (after relations confirmed) --
if st.session_state.get("ontology_generated") and st.session_state.get("relation_review_done"):
    structure = st.session_state["structure"]
    syn_map = st.session_state.get("synonyms_map", {})
    # Step 4: Load base ontology (EMMO/MatGraph) if available
    base_graph = None
    base_urls = [
        "https://raw.githubusercontent.com/MaxDreger92/MatGraph/enhancement/publication/Ontology/MatGraphOntology.ttl",
        "https://raw.githubusercontent.com/MaxDreger92/MatGraph/master/Ontology/MatGraphOntology.ttl"
    ]
    with st.spinner("Loading base ontology (EMMO/MatGraph core)..."):
        for url in base_urls:
            try:
                res = requests.get(url, timeout=10)
                if res.status_code == 200:
                    base_graph = Graph()
                    base_graph.parse(data=res.content, format="turtle")
                    st.info(f"Base ontology loaded from `{url}`")
                    break
            except Exception:
                continue
        if not base_graph:
            st.warning("Base ontology could not be retrieved. A minimal core ontology will be used.")
    # Step 5: Assemble ontology graph
    with st.spinner("Building ontology graph..."):
        G, base_ns = assemble_ontology_graph(structure, base_graph)
    # Step 6: External ontology alignment
    class_cat_pairs: List[tuple] = []
    for cat_key in ["materials", "properties", "parameters", "manufacturings", "measurements", "simulations", "metadata"]:
        for name in structure.get(cat_key, []):
            cat_label = cat_key[:-1] if cat_key.endswith('s') else cat_key
            if cat_label == "manufacturing": cat_label = "material process"
            if cat_label == "measurement": cat_label = "measurement process"
            if cat_label == "simulation": cat_label = "simulation process"
            class_cat_pairs.append((name, cat_label.capitalize()))
    align_map = {}
    if class_cat_pairs:
        with st.spinner("Aligning classes with external ontologies (QUDT, ChEBI, EMMO, etc.)..."):
            align_map = map_to_external(class_cat_pairs)
            add_external_mappings(G, align_map, base_ns)
    if align_map:
        aligned_list = [f"{cls} ? *{info.get('ontology', '')}* (`{info.get('uri', '')}`)" for cls, info in align_map.items()]
        st.write("**External Alignments:** Mapped classes to external ontologies:")
        st.write(", ".join(aligned_list))
    else:
        st.write("**External Alignments:** No external ontology mappings were found for the given classes.")
    # Step 7: Add labels (prefLabel, altLabel) and definitions for classes
    with st.spinner("Adding labels and definitions to classes..."):
        add_labels_and_definitions(G, syn_map, structure, base_ns)
    # Step 8: SHACL validation (domain/range checks for relations)
    shacl_report = run_shacl_validation(G, base_ns)
    if shacl_report:
        if shacl_report.startswith("SKIPPED"):
            st.warning("SHACL validation was skipped (pyshacl library not installed).")
        else:
            md_box("### ? SHACL Validation Failed", "error")
            st.text(shacl_report)
            st.stop()
    else:
        st.success("SHACL validation passed ?")
    # Step 9: Apply versioning metadata and finalize graph
    apply_versioning(G, base_graph)
    # Step 10: Serialize and offer downloads
    try:
        ttl_data = G.serialize(format="turtle")
    except Exception as e:
        st.error(f"Error serializing ontology to Turtle: {e}")
        ttl_data = None
    if ttl_data:
        st.download_button("?? Download Ontology (Turtle)", ttl_data, "hydrogen_ontology.ttl", mime="text/turtle")
    # NetworkX Graph for GraphML and visualization
    nxG = nx.MultiDiGraph()
    base_nodes = ["Material", "Property", "Parameter", "Process"]
    if structure.get("metadata"): base_nodes.append("Metadata")
    if structure.get("manufacturings"): base_nodes.append("Manufacturing")
    if structure.get("measurements"): base_nodes.append("Measurement")
    if structure.get("simulations"): base_nodes.append("Simulation")
    # Add base category nodes
    for bn in base_nodes:
        nxG.add_node(bn, category=bn, is_base=True)
    # Add class nodes and subclass-of edges
    for key, parent in [("materials", "Material"), ("properties", "Property"), ("parameters", "Parameter"),
                        ("manufacturings", "Manufacturing" if structure.get("manufacturings") else "Process"),
                        ("measurements", "Measurement" if structure.get("measurements") else "Process"),
                        ("simulations", "Simulation" if structure.get("simulations") else "Process"),
                        ("metadata", "Metadata")]:
        for item in structure.get(key, []):
            node = to_valid_uri(item)
            nxG.add_node(node, category=parent, is_base=False)
            if parent and parent not in ["Metadata"]:
                nxG.add_edge(node, parent, label="isA")  # subclass relationship
    # Add relationship edges (hasProperty, influences)
    for rel in structure.get("relationships", []):
        subj = to_valid_uri(rel["subject"]); pred = rel["predicate"]; obj = to_valid_uri(rel["object"])
        # Ensure nodes exist
        if subj not in nxG:
            nxG.add_node(subj, category="", is_base=False)
        if obj not in nxG:
            nxG.add_node(obj, category="", is_base=False)
        nxG.add_edge(subj, obj, label=pred)
    # Provide GraphML download
    try:
        from io import BytesIO
        graphml_buffer = BytesIO()
        nx.write_graphml(nxG, graphml_buffer, encoding="utf-8")
        graphml_data = graphml_buffer.getvalue().decode("utf-8")
        st.download_button("?? Download Graph (GraphML)", graphml_data, "hydrogen_ontology.graphml", mime="application/xml")
    except Exception as e:
        st.error(f"Error generating GraphML: {e}")
    # Visualize graph using Graphviz
    try:
        st.subheader("Ontology Graph Visualization")
        dot = Digraph(engine="dot", graph_attr={"rankdir": "TB"})
        # Node styling by category
        category_shape = {
            "Material": "box",
            "Property": "ellipse",
            "Parameter": "diamond",
            "Process": "polygon",
            "Manufacturing": "polygon",
            "Measurement": "polygon",
            "Simulation": "polygon",
            "Metadata": "note",
            "": "oval"
        }
        category_color = {
            "Material": "#ffdede",      # light red
            "Property": "#deffde",      # light green
            "Parameter": "#dedeff",     # light blue
            "Process": "#fff2cc",       # light yellow
            "Manufacturing": "#fff2cc", # same family as Process
            "Measurement": "#fff2cc",
            "Simulation": "#fff2cc",
            "Metadata": "#f0f0f0",      # light grey
            "": "#ffffff"
        }
        # Add nodes with style
        for node, data in nxG.nodes(data=True):
            cat = data.get("category", "")
            is_base = data.get("is_base", False)
            shape = category_shape.get(cat, "oval")
            style = "filled" if is_base else "solid"
            fillcolor = category_color.get(cat, "#ffffff") if is_base else "#ffffff"
            # Use label with spaces for CamelCase classes (not for base categories)
            label = node
            if not is_base:
                label = CAMEL_RE.sub(' ', node)
            dot.node(node, label, shape=shape, style=style, fillcolor=fillcolor)
        # Add edges with labels
        for u, v, edata in nxG.edges(data=True):
            edge_label = edata.get("label", "")
            dot.edge(u, v, label=edge_label)
        st.graphviz_chart(dot.source)
    except Exception as e:
        st.error(f"Graph visualization failed: {e}")
