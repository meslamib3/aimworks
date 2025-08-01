import streamlit as st
import os, re, json, datetime
from typing import List, Dict, Optional

# LLM and multi-agent orchestration libraries
try:
    import openai
    from crewai import Agent, Task, Crew, Process
    from crewai.llm import LLM
except ImportError as e:
    st.error(f"Missing library: {e.name}. Please install it to use this app.")
    st.stop()

# Semantic web and graph libraries
try:
    from rdflib import Graph, Namespace, RDF, RDFS, OWL, XSD, Literal, URIRef
    import networkx as nx
    from graphviz import Digraph
    import requests
except ImportError as e:
    st.error(f"Missing dependency: {e.name}. Install it and retry.")
    st.stop()

# ---- Streamlit UI ----
st.set_page_config(page_title="Hydrogen Ontology Generator", page_icon="??", layout="centered")
st.title("Hydrogen Technology Ontology Generator (EMMO-based)")
st.write("**Description:** Provide a hydrogen technology research question to automatically generate an extended ontology. The system uses multiple AI agents (feature scientist, terminology expert, ontology engineer, etc.) to identify relevant concepts, unify synonyms, align with external vocabularies (QUDT, ChEBI, PROV, etc.), and build a rich OWL ontology. An interactive loop allows you to confirm or adjust synonym merges or other ambiguities. The final ontology (TTL) and a graph visualization are provided for download or inspection.")

# User inputs
question: str = st.text_input("**Research Question**:", help="Enter a specific research question in the hydrogen technology domain (e.g., about hydrogen storage materials, membrane experiments, etc.)")
api_key: str = st.text_input("**OpenAI API Key** (required for GPT-4 agents):", type="password", help="Your OpenAI API key for GPT-4 access. (Needed to run the AI agents)")

# Session state initialization
if "features_text" not in st.session_state:
    st.session_state["features_text"] = ""
if "last_ont_uri" not in st.session_state:
    st.session_state["last_ont_uri"] = None

# Feature suggestion agent
def suggest_features(question_text: str) -> str:
    """Use a specialized agent to suggest relevant features for the given question."""
    if not question_text:
        return ""
    # Configure OpenAI LLM
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    llm = LLM(model="openai/gpt-4", temperature=0.2, max_tokens=512)
    scientist = Agent(
        role="Hydrogen-Feature Scientist",
        goal="Identify key experimental, material, and environmental features in hydrogen technology relevant to the research question.",
        backstory=("You are a veteran hydrogen technology researcher, highly familiar with materials, processes, and parameters in hydrogen storage and fuel cell experiments. "
                   "You excel at concisely naming important features (in CamelCase) involved in any given hydrogen technology research problem."),
        llm=llm
    )
    prompt = (
        "Analyze the research question below and list 5-15 relevant features (nouns or noun phrases) involved in experiments or data for this topic. "
        "These may include materials, properties, parameters/conditions, processes (manufacturing or measurement), etc. "
        "Provide the features as a single comma-separated list (no explanations, no line breaks). Use CamelCase for multi-word terms (e.g. HydrogenPermeability, CatalystInk) for consistency.\n"
        f"**Research Question:** \"{question_text}\""
    )
    task = Task(description=prompt, expected_output="Comma-separated feature list", agent=scientist)
    Crew(agents=[scientist], tasks=[task], process=Process.sequential).kickoff()
    output = str(task.output).strip().strip("`")  # remove markdown formatting if any
    return output

# "Suggest Features" button logic
if st.button("Suggest Features"):
    if not api_key:
        st.error("Please provide an OpenAI API key to use GPT-4 for feature suggestion.")
    else:
        os.environ["OPENAI_API_KEY"] = api_key
        with st.spinner("AI Scientist identifying relevant features..."):
            feature_list = suggest_features(question)
        if feature_list:
            st.session_state["features_text"] = feature_list
            st.success("Feature suggestions generated. You may review/edit them below.")
        else:
            st.warning("No features suggested (the question might be too broad/narrow). Please refine your question or add features manually.")

# Feature list text area (editable by user)
features_text: str = st.text_area("**Identified Features** (comma-separated):",
                                  value=st.session_state.get("features_text", ""),
                                  height=100,
                                  help="List of features to include in the ontology. You can edit this list (add/remove/rename features) before generating the ontology.")
st.session_state["features_text"] = features_text  # sync state
features: List[str] = [f.strip() for f in features_text.split(",") if f.strip()]
if features and features_text and not question:
    st.info("Using provided features without a specific question. (A general ontology will be generated.)")

# Helper: normalize string to a valid URI fragment (CamelCase)
def to_valid_uri(name: str) -> str:
    # Remove non-alphanumeric characters, then CamelCase each word
    parts = re.split(r"[\W_]+", name)
    uri = "".join(word.capitalize() for word in parts if word)
    return uri or name

# Agents for synonyms and alignment
def find_synonyms(feature_list: List[str]) -> Dict[str, List[str]]:
    """Use an agent to find synonyms/aliases among the features."""
    if not feature_list:
        return {}
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    llm = LLM(model="openai/gpt-4", temperature=0.0, max_tokens=512)
    terminologist = Agent(
        role="Lexical Synonym Expert",
        goal="Identify which feature terms refer to the same concept and propose a unified naming.",
        backstory=("You are an ontology terminologist specialized in hydrogen domain jargon. You know common synonyms, abbreviations, and alternate names for concepts. "
                   "Your task is to merge equivalent terms."),
        llm=llm
    )
    feature_list_str = "; ".join(feature_list)
    syn_prompt = (
        "The following list contains feature terms that may include synonyms or abbreviations referring to the same concept. "
        "Group equivalent terms together and choose one canonical name per group. Use domain knowledge (hydrogen technology) to recognize synonyms (including symbols or acronyms). "
        "Output JSON mapping each canonical term to a list of its alternate terms (synonyms or abbreviations). "
        "If a term has no synonyms in the list, map it to an empty list.\n"
        f"Features: {feature_list_str}"
    )
    task = Task(description=syn_prompt, expected_output="JSON {canonical: [synonyms,...]}", agent=terminologist)
    Crew(agents=[terminologist], tasks=[task], process=Process.sequential).kickoff()
    raw = str(task.output).strip().strip("`")
    # Attempt JSON parse, with minor fixes if needed
    try:
        syn_map = json.loads(raw)
    except json.JSONDecodeError:
        try:
            syn_map = json.loads(re.sub(r"\'", "\"", raw))
        except Exception:
            syn_map = {}
    # Ensure keys and values are lists of strings
    cleaned_map: Dict[str, List[str]] = {}
    if isinstance(syn_map, dict):
        for canon, alts in syn_map.items():
            if isinstance(canon, str):
                if isinstance(alts, list):
                    cleaned_map[canon.strip()] = [str(a).strip() for a in alts if isinstance(a, str)]
                else:
                    cleaned_map[canon.strip()] = []
    return cleaned_map

def map_to_external(classes: List[tuple]) -> Dict[str, Dict[str, str]]:
    """Use an agent to align classes to external vocabularies (QUDT, ChEBI, EMMO, etc.).
    `classes` is a list of (name, category) for each class."""
    mapping: Dict[str, Dict[str, str]] = {}
    if not classes:
        return mapping
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    llm = LLM(model="openai/gpt-4", temperature=0.0, max_tokens=600)
    aligner = Agent(
        role="Ontology Alignment Specialist",
        goal="Find equivalent concepts in established ontologies for given terms.",
        backstory=("You are an expert in semantic web standards (QUDT, ChEBI, EMMO, etc.) and know common ontology entries for chemical substances, units, and physical quantities. "
                   "You map local ontology classes to standard references."),
        llm=llm
    )
    # Prepare class list with category hints
    classes_str = "\n".join([f"- {name} ({cat})" for name, cat in classes])
    align_prompt = (
        "For each of the following ontology classes (with their category), find if there is an equivalent concept in a well-known external ontology/vocabulary. "
        "Use: ChEBI for chemical substances/materials, QUDT for physical quantities or units, EMMO or other domain ontologies for common concepts if applicable. "
        "If a match is found, provide the ontology name and the precise URI of the equivalent class. If no well-known ontology has that concept, omit it.\n"
        "Output JSON where each input class name maps to an object with 'ontology' and 'uri' keys (or provide an empty JSON if no mappings found)."
        "\nClasses:\n" + classes_str
    )
    task = Task(description=align_prompt, expected_output="JSON mapping classes to {ontology, uri}", agent=aligner)
    Crew(agents=[aligner], tasks=[task], process=Process.sequential).kickoff()
    raw = str(task.output).strip().strip("`")
    try:
        mapping = json.loads(raw)
    except json.JSONDecodeError:
        try:
            mapping = json.loads(raw.replace("'", "\""))
        except Exception:
            mapping = {}
    final_map = {}
    if isinstance(mapping, dict):
        for cls, info in mapping.items():
            if isinstance(info, dict):
                ont = str(info.get("ontology", "")).strip()
                uri = str(info.get("uri", "")).strip()
                if uri:
                    final_map[cls.strip()] = {"ontology": ont, "uri": uri}
    return final_map

# Ontology construction
def build_ontology_structure(question_text: str, feature_list: List[str]) -> Optional[dict]:
    """Runs the multi-agent reasoning to classify features and suggest relationships, returning a structured ontology dict."""
    if not feature_list:
        return None
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    # Agents: Domain expert (classify features & propose relations), Ontology engineer (produce structured JSON)
    llm_reasoner = LLM(model="openai/gpt-4", temperature=0.3, max_tokens=1200)
    llm_formatter = LLM(model="openai/gpt-4", temperature=0.0, max_tokens=1000)
    domain_agent = Agent(
        role="Hydrogen Domain Expert",
        goal="Classify features and propose semantic relationships",
        backstory=("You are a hydrogen technology domain expert and ontologist. You accurately identify each feature's type (Material, Property, Parameter, Manufacturing, Measurement, Simulation, or Metadata) and how they relate."),
        llm=llm_reasoner
    )
    engineer_agent = Agent(
        role="Ontology Engineer",
        goal="Convert expert analysis into structured ontology data (classes and relationships)",
        backstory=("You take the domain expert's analysis and produce a clean, valid JSON representation of ontology classes and relations."),
        llm=llm_formatter
    )
    feat_display = "; ".join(feature_list)
    classification_prompt = (
        f"Research Question: {question_text or '(general context)'}\n"
        f"Features: {feat_display}\n\n"
        "For each feature, determine its category:\n"
        "- **Material**: a physical substance or component (e.g., a chemical, material, or device)\n"
        "- **Property**: a measurable property or quantity (e.g., permeability, pressure, conductivity)\n"
        "- **Parameter**: an experimental or environmental condition/parameter (e.g., temperature, pressure setpoint, duration)\n"
        "- **Manufacturing**: a fabrication or treatment process (e.g., HeatingStep, SinteringProcess)\n"
        "- **Measurement**: a measurement or characterization method/process (e.g., PressureTest, SEMImaging)\n"
        "- **Simulation**: a computational modeling or simulation activity (e.g., CFDModel, DFTCalculation)\n"
        "- **Metadata**: contextual or identifier info (e.g., SampleID, Date, Operator) not part of the domain model but supporting data management\n\n"
        "Next, suggest meaningful relationships:\n"
        "- Use **hasProperty** (Material ? Property) for a material possessing a property (e.g., a membrane hasProperty hydrogenPermeability).\n"
        "- Use **influences** (Parameter/Process ? Property) for a parameter or process affecting a property (e.g., Pressure influences PermeationRate, or AnnealingProcess influences MembraneStrength).\n"
        "(Only use these two relations; do NOT introduce others here. Do not relate Material to Material or similar, focus on Parameter/Process to Property influences and Material-property assignments.)\n\n"
        "Finally, list any such relationships clearly. If none are obvious for a feature, you may omit it from relationships.\n"
        "Be concise and clear."
    )
    format_prompt = (
        "Now convert the analysis into JSON with keys: 'materials', 'properties', 'parameters', 'manufacturings', 'measurements', 'simulations', 'metadata', and 'relationships'. "
        "'materials', 'properties', etc. should each be a list of feature names in that category. "
        "'relationships' should be a list of objects with 'subject', 'predicate', 'object' (using exactly 'hasProperty' or 'influences'). "
        "Exclude any feature from categories if it was merged with another (only list unique concepts once). "
        "Ensure every feature appears in exactly one category list. Use CamelCase identifiers as provided. "
        "Only output valid JSON, no extra text."
    )
    t1 = Task(description=classification_prompt, expected_output="Feature classifications and relationships (text)", agent=domain_agent)
    t2 = Task(description=format_prompt, expected_output="JSON ontology structure", agent=engineer_agent, context=[t1])
    Crew(agents=[domain_agent, engineer_agent], tasks=[t1, t2], process=Process.sequential).kickoff()
    raw_output = str(t2.output)
    # Clean up any Markdown formatting or extra text around JSON
    raw_output = raw_output.strip().lstrip("```").rstrip("```")
    # Extract JSON content between the first '{' and the last '}'
    start_idx = raw_output.find('{')
    end_idx = raw_output.rfind('}')
    if start_idx != -1 and end_idx != -1:
        raw_json_str = raw_output[start_idx:end_idx+1]
    else:
        raw_json_str = raw_output
    raw_json_str = raw_json_str.strip()
    try:
        structure = json.loads(raw_json_str)
    except json.JSONDecodeError:
        try:
            # Try replacing single quotes with double quotes
            structure = json.loads(raw_json_str.replace("'", "\""))
        except json.JSONDecodeError:
            st.error("Ontology structuring failed. The agent output was not valid JSON.")
            st.text(raw_output)  # Show raw output for debugging
            return None
    # Validate that all provided features are categorized
    input_features_set = set(feature_list)
    categorized_set = set()
    for cat_key in ["materials", "properties", "parameters", "manufacturings", "measurements", "simulations", "metadata"]:
        for item in structure.get(cat_key, []):
            categorized_set.add(item)
    missing = input_features_set - categorized_set
    if missing:
        st.warning(f"The following features were not categorized by the AI and will be added as 'Metadata': {', '.join(missing)}")
        structure.setdefault("metadata", [])
        for item in missing:
            if item not in structure["metadata"]:
                structure["metadata"].append(item)
    # Remove any duplicate entries in category lists
    for cat_key in ["materials", "properties", "parameters", "manufacturings", "measurements", "simulations", "metadata"]:
        if cat_key in structure and isinstance(structure[cat_key], list):
            unique_list = []
            seen = set()
            for val in structure[cat_key]:
                if val not in seen:
                    unique_list.append(val)
                    seen.add(val)
            structure[cat_key] = unique_list
    return structure

def assemble_ontology_graph(structure: dict, base_graph: Optional[Graph]) -> (Graph, Namespace):
    """Construct the extended ontology (rdflib Graph) given the structured data and an optional base ontology Graph. Returns the graph and the base namespace used."""
    G = Graph()
    # If base ontology provided, include it
    if base_graph:
        G += base_graph
    # Determine base namespace for new classes
    base_ns = None
    if base_graph:
        for subj in base_graph.subjects(RDF.type, OWL.Ontology):
            base_ns = Namespace(str(subj) + "#")
            break
    if not base_ns:
        base_ns = Namespace("http://example.org/matgraph#")
    # Bind prefixes
    G.bind("mat", base_ns)
    G.bind("owl", OWL); G.bind("rdf", RDF); G.bind("rdfs", RDFS); G.bind("xsd", XSD)
    G.bind("skos", Namespace("http://www.w3.org/2004/02/skos/core#"))
    G.bind("prov", Namespace("http://www.w3.org/ns/prov#"))
    G.bind("qudt", Namespace("http://qudt.org/schema/qudt/"))
    G.bind("quantitykind", Namespace("http://qudt.org/vocab/quantitykind/"))
    G.bind("chebi", Namespace("http://purl.obolibrary.org/obo/CHEBI_"))
    # Helper to ensure a class exists
    def ensure_class(uri: URIRef):
        if (uri, RDF.type, OWL.Class) not in G:
            G.add((uri, RDF.type, OWL.Class))
    # Ensure core ontology classes exist
    core_classes = {
        "Material": base_ns.Material,
        "Property": base_ns.Property,
        "Parameter": base_ns.Parameter,
        "Process": base_ns.Process,
        "Metadata": base_ns.Metadata
    }
    for cname, uri in core_classes.items():
        ensure_class(uri)
    # Define subclasses for specific process types if present
    if structure.get("manufacturings") or structure.get("measurements") or structure.get("simulations"):
        ensure_class(base_ns.Process)
    if structure.get("manufacturings"):
        ensure_class(base_ns.Manufacturing)
        G.add((base_ns.Manufacturing, RDFS.subClassOf, base_ns.Process))
    if structure.get("measurements"):
        ensure_class(base_ns.Measurement)
        G.add((base_ns.Measurement, RDFS.subClassOf, base_ns.Process))
    if structure.get("simulations"):
        ensure_class(base_ns.Simulation)
        G.add((base_ns.Simulation, RDFS.subClassOf, base_ns.Process))
    # Add feature classes as subclasses of their category
    category_mapping = {
        "materials": base_ns.Material,
        "properties": base_ns.Property,
        "parameters": base_ns.Parameter,
        "manufacturings": base_ns.Manufacturing if structure.get("manufacturings") else base_ns.Process,
        "measurements": base_ns.Measurement if structure.get("measurements") else base_ns.Process,
        "simulations": base_ns.Simulation if structure.get("simulations") else base_ns.Process,
        "metadata": None
    }
    for key, parent in category_mapping.items():
        for item in structure.get(key, []):
            class_name = to_valid_uri(item)
            class_uri = base_ns[class_name]
            ensure_class(class_uri)
            if parent and key != "metadata":
                G.add((class_uri, RDFS.subClassOf, parent))
    # Add object properties if not already present
    hasP = base_ns.hasProperty
    infl = base_ns.influences
    if (hasP, RDF.type, OWL.ObjectProperty) not in G:
        G.add((hasP, RDF.type, OWL.ObjectProperty))
        G.add((hasP, RDFS.domain, base_ns.Material))
        G.add((hasP, RDFS.range, base_ns.Property))
        G.add((hasP, RDFS.label, Literal("has property", lang="en")))
    if (infl, RDF.type, OWL.ObjectProperty) not in G:
        G.add((infl, RDF.type, OWL.ObjectProperty))
        G.add((infl, RDFS.domain, base_ns.Process))
        G.add((infl, RDFS.range, base_ns.Property))
        G.add((infl, RDFS.label, Literal("influences", lang="en")))
    # Add extended schema data properties and PROV alignment
    val_prop = base_ns.hasValue
    unit_prop = base_ns.hasUnit
    if (val_prop, RDF.type, OWL.DatatypeProperty) not in G:
        G.add((val_prop, RDF.type, OWL.DatatypeProperty))
        G.add((val_prop, RDFS.domain, base_ns.Property))
        G.add((val_prop, RDFS.range, XSD.double))
        G.add((val_prop, RDFS.label, Literal("has value", lang="en")))
    if (unit_prop, RDF.type, OWL.ObjectProperty) not in G:
        G.add((unit_prop, RDF.type, OWL.ObjectProperty))
        G.add((unit_prop, RDFS.domain, base_ns.Property))
        G.add((unit_prop, RDFS.range, URIRef("http://qudt.org/schema/qudt/Unit")))
        G.add((unit_prop, RDFS.label, Literal("has unit", lang="en")))
    exp_class = base_ns.Experiment
    if (exp_class, RDF.type, OWL.Class) not in G:
        G.add((exp_class, RDF.type, OWL.Class))
        G.add((exp_class, RDFS.subClassOf, URIRef("http://www.w3.org/ns/prov#Activity")))
        G.add((exp_class, RDFS.label, Literal("Experiment", lang="en")))
    # Add relationships from structure
    seen_rels = set()
    for rel in structure.get("relationships", []):
        subj = rel.get("subject"); pred = rel.get("predicate"); obj = rel.get("object")
        if not subj or not pred or not obj:
            continue
        subj_uri = base_ns[to_valid_uri(subj)]
        obj_uri = base_ns[to_valid_uri(obj)]
        if pred == "hasProperty":
            pred_uri = base_ns.hasProperty
        elif pred == "influences":
            pred_uri = base_ns.influences
        else:
            pred_uri = None
        if pred_uri is None or subj_uri == obj_uri:
            continue  # skip invalid or self-relations
        triple = (subj_uri, pred_uri, obj_uri)
        if triple in seen_rels:
            continue
        seen_rels.add(triple)
        G.add(triple)
    return G, base_ns

def add_external_mappings(G: Graph, mapping: Dict[str, Dict[str, str]], base_ns: Namespace) -> None:
    """Add owl:equivalentClass links for any external ontology mappings provided."""
    for cls_name, info in mapping.items():
        ext_uri_str = info.get("uri")
        if not ext_uri_str:
            continue
        cls_uri = None
        # Determine class URI in our ontology for this class name
        cls_candidate = base_ns[to_valid_uri(cls_name)]
        if (cls_candidate, RDF.type, OWL.Class) in G:
            cls_uri = cls_candidate
        if not cls_uri:
            continue
        ext_uri = URIRef(ext_uri_str)
        # Bind known external ontology prefixes if applicable
        ont_name = str(info.get("ontology", "")).lower()
        if ont_name.startswith("chebi") or "chebi" in ext_uri_str.lower():
            G.bind("chebi", "http://purl.obolibrary.org/obo/CHEBI_")
            if ext_uri_str.startswith("CHEBI:"):
                chebi_id = ext_uri_str.split(":")[1]
                ext_uri = URIRef(f"http://purl.obolibrary.org/obo/CHEBI_{chebi_id}")
        elif ont_name.startswith("qudt") or "quantitykind" in ext_uri_str.lower():
            G.bind("quantitykind", "http://qudt.org/vocab/quantitykind/")
        elif ont_name.startswith("emmo"):
            G.bind("emmo", "http://emmo.info/emmo#")
        # Add equivalentClass mapping and an annotation
        G.add((cls_uri, OWL.equivalentClass, ext_uri))
        G.add((cls_uri, RDFS.comment, Literal(f"Equivalent to {info.get('ontology')} term {info.get('uri')}", lang="en")))

def add_labels_and_definitions(G: Graph, synonyms_map: Dict[str, List[str]], structure: dict, base_ns: Namespace) -> None:
    """Add lexical labels (skos:prefLabel, skos:altLabel) and brief definitions to classes in the graph."""
    classes_with_cats: List[tuple] = []
    for key in ["materials", "properties", "parameters", "manufacturings", "measurements", "simulations", "metadata"]:
        for item in structure.get(key, []):
            cat = key[:-1] if key.endswith("s") else key
            if cat == "manufacturing": cat = "manufacturing process"
            if cat == "measurement": cat = "measurement process"
            classes_with_cats.append((item, cat.capitalize()))
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    def_agent = Agent(
        role="Ontology Documentation Specialist",
        goal="Provide concise definitions for ontology classes",
        backstory=("You are a scientist who succinctly defines technical terms in hydrogen technology for ontology documentation. "
                   "Your definitions are accurate and 1-2 sentences long."),
        llm=LLM(model="openai/gpt-4", temperature=0.3, max_tokens=100)
    )
    for class_name, category in classes_with_cats:
        cls_uri = base_ns[to_valid_uri(class_name)]
        if (cls_uri, RDF.type, OWL.Class) not in G:
            continue
        # Add preferred label (CamelCase split into words if needed)
        pref_label = class_name.strip()
        if re.search(r'[a-z][A-Z]', pref_label):
            pref_label = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', pref_label)
        G.add((cls_uri, Namespace("http://www.w3.org/2004/02/skos/core#").prefLabel, Literal(pref_label, lang="en")))
        # Add alternate labels for synonyms
        if class_name in synonyms_map:
            for alt in synonyms_map[class_name]:
                alt_label = alt.strip()
                if alt_label:
                    if re.search(r'[a-z][A-Z]', alt_label):
                        alt_label = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', alt_label)
                    G.add((cls_uri, Namespace("http://www.w3.org/2004/02/skos/core#").altLabel, Literal(alt_label, lang="en")))
        # Add a brief definition as comment if none exists yet
        if (cls_uri, RDFS.comment, None) not in G:
            definition_prompt = f"Provide a brief definition for '{class_name}' as a {category} in the context of hydrogen technology."
            task = Task(description=definition_prompt, expected_output="One-sentence definition.", agent=def_agent)
            Crew(agents=[def_agent], tasks=[task], process=Process.sequential).kickoff()
            def_text = str(task.output).strip().strip("`").replace('\n', ' ')
            if def_text:
                if def_text[-1] not in '.!?':
                    def_text += '.'
                G.add((cls_uri, RDFS.comment, Literal(def_text, lang="en")))

def apply_versioning(G: Graph, base_graph: Optional[Graph]) -> None:
    """Add ontology metadata: version info, imports, etc., and update session state for versioning."""
    ont_uri = URIRef("http://example.org/matgraph-ext")
    if base_graph:
        for subj in base_graph.subjects(RDF.type, OWL.Ontology):
            ont_uri = URIRef(str(subj) + "_extended")
            G.add((ont_uri, OWL.imports, subj))
            break
    G.add((ont_uri, RDF.type, OWL.Ontology))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    version_str = f"Generated on {timestamp}"
    G.add((ont_uri, OWL.versionInfo, Literal(version_str, datatype=XSD.string)))
    if st.session_state.get("last_ont_uri"):
        prev_uri = URIRef(st.session_state["last_ont_uri"])
        G.add((ont_uri, OWL.priorVersion, prev_uri))
    st.session_state["last_ont_uri"] = str(ont_uri)

# Main "Generate Ontology" pipeline
if st.button("Generate Ontology"):
    if not api_key:
        st.error("Please provide your OpenAI API key to generate the ontology (GPT-4 is required).")
        st.stop()
    if not features:
        st.error("Please enter or confirm at least one feature in the list.")
        st.stop()
    os.environ["OPENAI_API_KEY"] = api_key

    # 1. Fetch base ontology (MatGraph + EMMO) from GitHub or use minimal core
    base_graph = None
    base_url_primary = "https://raw.githubusercontent.com/MaxDreger92/MatGraph/enhancement/publication/Ontology/MatGraphOntology.ttl"
    base_url_alt = "https://raw.githubusercontent.com/MaxDreger92/MatGraph/master/Ontology/MatGraphOntology.ttl"
    with st.spinner("Fetching base ontology (EMMO/MatGraph) ..."):
        for url in [base_url_primary, base_url_alt]:
            try:
                res = requests.get(url, timeout=10)
                if res.status_code == 200:
                    base_graph = Graph()
                    base_graph.parse(data=res.content, format="turtle")
                    st.info(f"Base ontology loaded from {url}")
                    break
            except Exception:
                continue
    if not base_graph:
        st.warning("Base ontology could not be retrieved. The ontology will be generated using a minimal core schema.")
    # 2. Synonym unification
    syn_map = {}
    if features:
        with st.spinner("Analyzing synonyms and jargon..."):
            syn_map = find_synonyms(features)
    if syn_map:
        merged_features = []
        merged_info = []
        for canonical, alts in syn_map.items():
            canonical = canonical.strip()
            if canonical:
                merged_features.append(canonical)
                if alts:
                    merged_info.append(f"**{canonical}** (aliases: {', '.join(alts)})")
        if merged_info:
            st.write("**Synonym Merging:** The following terms were identified as referring to the same concept and will be merged into one class:")
            st.write("; ".join(merged_info))
            st.info("If any merge is incorrect, please adjust the feature list and regenerate.")
        features = merged_features
    # 3. Ontology structure generation
    with st.spinner("Classifying features and extracting relationships via AI..."):
        structure = build_ontology_structure(question, features)
    if not structure:
        st.error("Ontology generation failed at the structuring stage. Please try again with different input or check logs.")
        st.stop()
    # 4. Build ontology graph
    with st.spinner("Constructing ontology graph..."):
        G, base_ns = assemble_ontology_graph(structure, base_graph)
    # 5. External vocabulary alignment
    align_map = {}
    with st.spinner("Aligning with external ontologies (QUDT, ChEBI, etc.)..."):
        class_cat_list = []
        for cat_key in ["materials", "properties", "parameters", "manufacturings", "measurements", "simulations", "metadata"]:
            for item in structure.get(cat_key, []):
                cat_name = cat_key[:-1] if cat_key.endswith('s') else cat_key
                if cat_name == "manufacturing": cat_name = "material process"
                if cat_name == "measurement": cat_name = "measurement process"
                class_cat_list.append((item, cat_name))
        align_map = map_to_external(class_cat_list)
        add_external_mappings(G, align_map, base_ns)
    if align_map:
        aligned_info = []
        for cls, info in align_map.items():
            aligned_info.append(f"{cls} ? *{info.get('ontology')}* (`{info.get('uri')}`)")
        st.write("**External Mappings:** Mapped some classes to external ontologies:")
        st.write(", ".join(aligned_info))
    else:
        st.write("**External Mappings:** No external ontology mappings were identified for these classes.")
    # 6. Add labels and definitions
    with st.spinner("Adding labels and definitions for classes..."):
        add_labels_and_definitions(G, syn_map, structure, base_ns)
    # 7. Apply versioning metadata
    apply_versioning(G, base_graph)
    # 8. Finalize and present results
    st.success("Ontology generation complete!")
    # Provide TTL download
    try:
        ttl_data = G.serialize(format="turtle")
        st.download_button("Download Ontology (TTL)", ttl_data, file_name="hydrogen_ontology.ttl", mime="text/turtle")
    except Exception as e:
        st.error(f"Error serializing ontology to Turtle: {e}")
    # Provide GraphML download
    try:
        nxG = nx.MultiDiGraph()
        base_nodes = ["Material", "Property", "Parameter", "Process"]
        if structure.get("metadata"):
            base_nodes.append("Metadata")
        if structure.get("manufacturings"):
            base_nodes.append("Manufacturing")
        if structure.get("measurements"):
            base_nodes.append("Measurement")
        if structure.get("simulations"):
            base_nodes.append("Simulation")
        for bn in base_nodes:
            nxG.add_node(bn, category=bn, is_base=1)
        for key, parent_cat in [("materials", "Material"), ("properties", "Property"), ("parameters", "Parameter"),
                                 ("manufacturings", "Manufacturing" if structure.get("manufacturings") else "Process"),
                                 ("measurements", "Measurement" if structure.get("measurements") else "Process"),
                                 ("simulations", "Simulation" if structure.get("simulations") else "Process"),
                                 ("metadata", "Metadata")]:
            for item in structure.get(key, []):
                node_name = to_valid_uri(item)
                nxG.add_node(node_name, category=parent_cat, is_base=0)
                if parent_cat and parent_cat not in ["Metadata"]:
                    nxG.add_edge(node_name, parent_cat, label="subClassOf")
        for subj_uri, pred_uri, obj_uri in G.triples((None, None, None)):
            if pred_uri == base_ns.hasProperty or pred_uri == base_ns.influences:
                subj_name = str(subj_uri).split("#")[-1]
                obj_name = str(obj_uri).split("#")[-1]
                if subj_name not in nxG:
                    nxG.add_node(subj_name, category="", is_base=0)
                if obj_name not in nxG:
                    nxG.add_node(obj_name, category="", is_base=0)
                edge_label = "hasProperty" if pred_uri == base_ns.hasProperty else "influences"
                nxG.add_edge(subj_name, obj_name, label=edge_label)
        graphml_data = nx.write_graphml_xml(nxG, encoding="utf-8")
        st.download_button("Download Graph (GraphML)", data=graphml_data, file_name="hydrogen_ontology.graphml", mime="application/xml")
    except Exception as e:
        st.error(f"Error exporting GraphML: {e}")
    # Display interactive graph visualization
    try:
        dot = Digraph(engine="dot", graph_attr={"rankdir": "TB"})
        category_shape = {
            "Material": "box",
            "Property": "ellipse",
            "Parameter": "diamond",
            "Process": "polygon",
            "Manufacturing": "polygon",
            "Measurement": "polygon",
            "Simulation": "polygon",
            "Metadata": "note",
            "": "ellipse"
        }
        category_color = {
            "Material": "lightgrey",
            "Property": "lightgrey",
            "Parameter": "lightgrey",
            "Process": "lightgrey",
            "Manufacturing": "lightgrey",
            "Measurement": "lightgrey",
            "Simulation": "lightgrey",
            "Metadata": "lightgrey",
            "": "white"
        }
        for node, data in nxG.nodes(data=True):
            cat = data.get("category", "")
            is_base = data.get("is_base", 0)
            shape = category_shape.get(cat, "ellipse")
            style = "filled" if is_base == 1 else ""
            fillcolor = category_color.get(cat, "white") if is_base == 1 else "white"
            dot.node(node, node, shape=shape, style=style, fillcolor=fillcolor)
        for u, v, edata in nxG.edges(data=True):
            label = edata.get("label", "")
            dot.edge(u, v, label=label)
        st.subheader("Ontology Graph Visualization")
        st.graphviz_chart(dot.source)
    except Exception as e:
        st.error(f"Graph visualization failed: {e}")
