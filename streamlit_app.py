# -*- coding: utf-8 -*-
# ------------------------------ hydrogen_ontology_generator.py ------------------------------
"""
Hydrogen Technology Ontology Generator (EMMO-based, aligned with RSC semantic model)

Full Streamlit app implementing:
  - Multi-agent pipeline (GPT-4 via CrewAI) for feature suggestion, synonym merging, ontology structuring, and critique
  - Extended semantic categories (Matter, Property, Parameter, Measurement, Manufacturing, Instrument, Agent, Data, Metadata, Identifier, Name, Value, Unit) aligned with EMMO/RSC models
  - Additional relations (hasName, hasIdentifier, measures, hasPart, isPartOf, wasAssociatedWith, hasOutputData, usesInstrument, hasInputMaterial, hasOutputMaterial, hasInputData, hasParameter, hasSubProcess, isSubProcessOf, hasValue, hasUnit) beyond core hasProperty/influences to capture names, identifiers, measurement links, inputs/outputs, subprocesses, values, and units
  - Integration of external ontologies (QUDT, ChEBI, EMMO, PROV-O) for alignment where possible
  - Option for user to review and edit relationships before finalizing
  - Output ontology in Turtle format and GraphML, with Graphviz visualization
"""
import streamlit as st
import os, re, json, datetime, textwrap, subprocess
from typing import List, Dict, Optional

# LLM & multi-agent orchestration (CrewAI)
try:
    import openai
except ImportError:
    subprocess.run(["pip", "install", "openai"], check=True)
    import openai
try:
    from crewai import Agent, Task, Crew, Process
    from crewai.llm import LLM
except ImportError:
    subprocess.run(["pip", "install", "crewai"], check=True)
    from crewai import Agent, Task, Crew, Process
    from crewai.llm import LLM

# Semantic Web stack (RDF, Graph)
try:
    from rdflib import Graph, Namespace, RDF, RDFS, OWL, XSD, Literal, URIRef, BNode
except ImportError:
    subprocess.run(["pip", "install", "rdflib"], check=True)
    from rdflib import Graph, Namespace, RDF, RDFS, OWL, XSD, Literal, URIRef, BNode
try:
    import networkx as nx
except ImportError:
    subprocess.run(["pip", "install", "networkx"], check=True)
    import networkx as nx
try:
    from graphviz import Digraph
except ImportError:
    subprocess.run(["pip", "install", "graphviz"], check=True)
    from graphviz import Digraph
try:
    import requests
except ImportError:
    subprocess.run(["pip", "install", "requests"], check=True)
    import requests

# Streamlit page configuration
st.set_page_config(page_title="Hydrogen Technology Ontology Generator (EMMO-based)", page_icon="??", layout="centered")
st.title("Hydrogen Technology Ontology Generator (EMMO-based)")

st.write(
    "**Description:** Provide a hydrogen technology research question (and optionally a list of features) to automatically generate an extended ontology. "
    "Multiple GPT-4 agents (feature suggestion, terminologist for synonyms, domain ontologist, etc.) collaborate to identify relevant concepts (categorized as Matter, Property, Parameter, Manufacturing, Measurement, Instrument, Agent, Data, Metadata, Identifier, Name, Value, Unit) and link them with meaningful relationships (hasProperty, influences, measures, hasName, hasIdentifier, hasPart, isPartOf, wasAssociatedWith, hasOutputData, usesInstrument, hasInputMaterial, hasOutputMaterial, hasInputData, hasParameter, hasSubProcess, isSubProcessOf, hasValue, hasUnit). "
    "A review step allows you to confirm or edit the proposed relationships before finalizing. "
    "The final ontology can be downloaded in Turtle or GraphML format, and a graph visualization is provided for inspection."
)

# User Inputs
question: str = st.text_input("**Research Question:**", 
    help="E.g. 'How does operating temperature affect hydrogen permeability of a Nafion membrane in PEM fuel cells?'")
api_key: str = st.text_input("**OpenAI API Key** (required for GPT-4 access):", type="password")

# Initialize session state variables
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
if "unit_replacements" not in st.session_state:
    st.session_state["unit_replacements"] = {}
if "last_ont_uri" not in st.session_state:
    st.session_state["last_ont_uri"] = None

# Utility functions
CAMEL_RE = re.compile(r'(?<=[a-z])(?=[A-Z])')

def to_valid_uri(label: str) -> str:
    """Convert an arbitrary label to a CamelCase string suitable for a URI fragment."""
    if not isinstance(label, str):
        label = str(label)
    label = label.strip()
    if not label:
        label = "UnnamedEntity"
    if label[0].isdigit():
        label = "V" + label
    parts = re.split(r"[\W_]+", label)
    valid_uri = "".join(word.capitalize() for word in parts if word)
    return valid_uri or "UnnamedEntity"

def md_box(msg: str, state: str = "info"):
    """Display a Markdown-styled message in a colored box (info, warning, error)."""
    getattr(st, state)(textwrap.dedent(msg).strip())

# Feature suggestion agent (Domain Expert)
def suggest_features(question_text: str) -> str:
    if not question_text.strip():
        return ""
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    llm = LLM(model="openai/gpt-4", temperature=0.2, max_tokens=512)
    domain_expert = Agent(
        role="Hydrogen Domain Expert",
        goal=(
            "Identify key materials, properties, parameters, and processes related to the given research question. "
            "Include typical units, value ranges, or measurement methods for each feature where applicable."
        ),
        backstory=(
            "You are a senior researcher in hydrogen technology. You know typical experimental conditions (temperatures, pressures, catalysts), materials (electrolytes, catalysts, membranes), performance metrics (efficiencies, yields), and processes (electrolysis, storage, degradation). "
            "Your task is to list the most relevant features for the research question in CamelCase, including units, ranges, or methods in parentheses."
        ),
        llm=llm
    )
    prompt = (
        "List **5-15** relevant features (materials, properties, parameters, processes, etc.) related to the research question below. "
        "Use CamelCase for each feature name, and after each name include in parentheses any typical units, common value ranges, or standard measurement methods for that feature (if applicable). "
        "Provide the list as a single comma-separated line, with no additional commentary beyond the parentheses content.\n\n"
        f"Research question: \"{question_text}\""
    )
    task = Task(description=prompt, expected_output="Comma-separated list of features with context in parentheses", agent=domain_expert)
    Crew(agents=[domain_expert], tasks=[task], process=Process.sequential).kickoff()
    output = str(task.output).strip().strip("`")
    return output.rstrip(", ")

# Synonym suggestion agent (Terminologist)
def find_synonyms(feature_list: List[str]) -> Dict[str, List[str]]:
    if not feature_list:
        return {}
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    llm = LLM(model="openai/gpt-4", temperature=0.0, max_tokens=500)
    terminologist = Agent(
        role="Lexical Terminologist",
        goal="Suggest synonyms or abbreviations for given hydrogen technology terms to enrich the ontology",
        backstory=(
            "You are an ontology terminologist familiar with hydrogen technology literature. "
            "Your task is to suggest common synonyms or abbreviations for each term."
        ),
        llm=llm
    )
    terms_str = "; ".join(feature_list)
    syn_prompt = (
        "For each term, list common synonyms or abbreviations that researchers use in scientific literature. "
        "Output a JSON object mapping each original term to a list of its synonyms/abbreviations. "
        "Include the term itself in the JSON with its synonyms. "
        "If no synonyms or abbreviations are known, use an empty list for that term.\n\n"
        f"Terms: {terms_str}"
    )
    task = Task(description=syn_prompt, expected_output='JSON mapping terms to synonym lists', agent=terminologist)
    Crew(agents=[terminologist], tasks=[task], process=Process.sequential).kickoff()
    raw = str(task.output).strip().strip("`")
    try:
        syn_map = json.loads(raw)
    except json.JSONDecodeError:
        # Attempt to correct minor formatting issues and parse again
        try:
            syn_map = json.loads(raw.replace("'", "\""))
        except json.JSONDecodeError:
            syn_map = {}
    final_map: Dict[str, List[str]] = {}
    if isinstance(syn_map, dict):
        for term, syns in syn_map.items():
            if isinstance(term, str):
                if isinstance(syns, list):
                    final_map[term.strip()] = [s.strip() for s in syns if isinstance(s, str)]
                else:
                    final_map[term.strip()] = []
    return final_map

# Ontology structuring agents (Domain Expert + Ontology Engineer)
def build_ontology_structure(question_text: str, feature_list: List[str]) -> Optional[Dict]:
    if not feature_list:
        return None
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    # Agent to analyze and categorize features, propose relations
    domain_llm = LLM(model="openai/gpt-4", temperature=0.3, max_tokens=1800)
    domain_agent = Agent(
        role="Hydrogen Domain Ontologist",
        goal="Categorize each feature and propose plausible relationships among them",
        backstory=(
            "You are an expert in hydrogen technology and ontology design. You will assign each feature to an ontology category and suggest scientifically plausible relationships using allowed predicates."
        ),
        llm=domain_llm
    )
    # Agent to format the output as structured JSON
    format_llm = LLM(model="openai/gpt-4", temperature=0.0, max_tokens=1200)
    engineer_agent = Agent(
        role="Ontology Engineer",
        goal="Format the ontology classes and relationships into a structured JSON object",
        backstory="You take the domain expert's analysis and convert it into a well-structured JSON representation of the ontology.",
        llm=format_llm
    )
    # Construct the prompt for domain_agent
    example_json = (
        "{\n"
        '  "matters": ["MembraneX"],\n'
        '  "properties": ["HydrogenPermeability"],\n'
        '  "parameters": ["TestingTemperature"],\n'
        '  "manufacturings": [],\n'
        '  "measurements": ["PermeabilityTest"],\n'
        '  "instruments": [],\n'
        '  "agents": [],\n'
        '  "data": [],\n'
        '  "metadata": [],\n'
        '  "identifiers": ["SampleID"],\n'
        '  "names": [],\n'
        '  "values": [],\n'
        '  "units": [],\n'
        '  "relationships": [\n'
        '    {"subject": "MembraneX", "predicate": "hasProperty", "object": "HydrogenPermeability"},\n'
        '    {"subject": "TestingTemperature", "predicate": "influences", "object": "HydrogenPermeability"},\n'
        '    {"subject": "PermeabilityTest", "predicate": "measures", "object": "HydrogenPermeability"},\n'
        '    {"subject": "MembraneX", "predicate": "hasIdentifier", "object": "SampleID"}\n'
        '  ]\n'
        '}'
    )
    analysis_prompt = (
        f"Research Question: {question_text or 'N/A'}\n"
        f"Features: {', '.join(feature_list)}\n\n"
        "Classify each feature into one of the following categories: "
        "Matter (material or substance), Property (material property or measurable quantity), Parameter (experimental condition or variable), "
        "Manufacturing (manufacturing or synthesis process), Measurement (measurement or testing process), Instrument (equipment or tool), "
        "Agent (person or organization), Data (dataset or result), Metadata (contextual information like conditions or descriptors), "
        "Identifier (ID or code), Name (textual name or label), Value (numeric value or constant), Unit (unit of measurement). "
        "Then propose plausible relationships among these features using only the following predicates:\n"
        "- **hasProperty**: (Matter -> Property) indicates a material has a property.\n"
        "- **influences**: (Parameter or Process -> Property) indicates a condition or process affects a property.\n"
        "- **measures**: (Measurement -> Property) indicates a measurement process measures a property.\n"
        "- **hasName**: (Entity -> Name) links an entity to a textual name or label.\n"
        "- **hasIdentifier**: (Entity -> Identifier) links an entity to an identifier or code.\n"
        "- **hasPart**: (Entity -> Entity) indicates one entity is a component or part of another.\n"
        "- **isPartOf**: (Entity -> Entity) indicates one entity is part of another (inverse of hasPart).\n"
        "- **wasAssociatedWith**: (Process -> Agent) indicates a process was carried out by or involved a certain agent (person or organization).\n"
        "- **hasOutputData**: (Process -> Data) indicates a process produced some data output.\n"
        "- **usesInstrument**: (Process -> Instrument) indicates a process uses a particular instrument or equipment.\n"
        "- **hasInputMaterial**: (Process -> Matter) indicates a process uses a material as input.\n"
        "- **hasOutputMaterial**: (Process -> Matter) indicates a process produces a material output.\n"
        "- **hasInputData**: (Process -> Data) indicates a process consumes some input data.\n"
        "- **hasParameter**: (Process -> Parameter) indicates a process or experiment has a controlled parameter.\n"
        "- **hasSubProcess**: (Process -> Process) indicates a process includes another as a subprocess.\n"
        "- **isSubProcessOf**: (Process -> Process) indicates a process is part of a larger process (inverse of hasSubProcess).\n"
        "- **hasValue**: (Parameter or Property -> Value) attaches a numeric value to a parameter or property.\n"
        "- **hasUnit**: (Parameter or Property -> Unit) specifies the unit of measure for a parameter or property.\n"
        "Only include relationships that make scientific sense. If unsure about a particular relation, omit it. "
        "Do NOT introduce any predicate not in the allowed list above.\n\n"
        "Finally, output the ontology as a JSON object with keys: "
        "'matters', 'properties', 'parameters', 'manufacturings', 'measurements', 'instruments', 'agents', 'data', 'metadata', 'identifiers', 'names', 'values', 'units', and 'relationships'. "
        "Each key should map to a list of feature names (use empty lists for categories with no entries). "
        "'relationships' should be a list of objects each with 'subject', 'predicate', 'object'.\n"
        "Example format:\n" + example_json
    )
    format_prompt = (
        "Format the ontology content as a JSON object with keys: "
        "'matters', 'properties', 'parameters', 'manufacturings', 'measurements', 'instruments', 'agents', 'data', 'metadata', 'identifiers', 'names', 'values', 'units', 'relationships'. "
        "Each of these keys should map to a list of terms (which may be empty). "
        "'relationships' should be a list of {\"subject\": ..., \"predicate\": ..., \"object\": ...} entries. "
        "Use only the allowed predicates and ensure the JSON is properly structured with no extra commentary."
    )
    t1 = Task(description=analysis_prompt, expected_output="Ontology analysis and categorization", agent=domain_agent)
    t2 = Task(description=format_prompt, expected_output="JSON ontology structure", agent=engineer_agent, context=[t1])
    Crew(agents=[domain_agent, engineer_agent], tasks=[t1, t2], process=Process.sequential).kickoff()
    raw_output = str(t2.output).strip().strip("`")
    # Extract JSON portion from output
    start_idx = raw_output.find("{")
    end_idx = raw_output.rfind("}")
    json_str = raw_output[start_idx: end_idx+1] if start_idx != -1 and end_idx != -1 else raw_output
    try:
        structure = json.loads(json_str)
    except json.JSONDecodeError:
        # Attempt minor fixes to JSON formatting and parse again
        fix_str = json_str.replace("'", "\"")
        fix_str = re.sub(r",\s*}", "}", fix_str)
        fix_str = re.sub(r",\s*\]", "]", fix_str)
        try:
            structure = json.loads(fix_str)
        except json.JSONDecodeError:
            # Try truncating last incomplete element if any
            trunc_idx = fix_str.rfind("},")
            if trunc_idx != -1:
                trunc_str = fix_str[:trunc_idx+1] + "]}"
                try:
                    structure = json.loads(trunc_str)
                    st.warning("Ontology JSON output was truncated. The last incomplete relation entry was removed to fix JSON format.")
                except Exception:
                    st.error("Failed to parse ontology JSON output from AI.")
                    st.text(raw_output)
                    return None
            else:
                st.error("Failed to parse ontology JSON output from AI.")
                st.text(raw_output)
                return None
    # Ensure all input features are categorized; if not, add to metadata
    input_set = set(feature_list)
    categorized_set = set()
    for cat in ["matters","properties","parameters","manufacturings","measurements","instruments","agents","data","metadata","identifiers","names","values","units"]:
        for item in structure.get(cat, []):
            categorized_set.add(item)
    missing = input_set - categorized_set
    if missing:
        structure.setdefault("metadata", [])
        for item in missing:
            if item not in structure["metadata"]:
                structure["metadata"].append(item)
        st.warning(f"The following features were not categorized and have been added as Metadata: {', '.join(missing)}")
    # Deduplicate entries in category lists
    for cat in ["matters","properties","parameters","manufacturings","measurements","instruments","agents","data","metadata","identifiers","names","values","units"]:
        if cat in structure and isinstance(structure[cat], list):
            structure[cat] = list(dict.fromkeys(structure[cat]))
    return structure

# Critic agent to review the drafted ontology structure
def critique_structure(structure: Dict) -> str:
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    llm = LLM(model="openai/gpt-4", temperature=0.0, max_tokens=400)
    critic = Agent(
        role="Ontology Critic",
        goal="Critique the proposed ontology structure for any implausible or missing relationships",
        backstory="You are a critical reviewer who checks the draft ontology for any relations that seem incorrect or any obvious missing relations.",
        llm=llm
    )
    critique_prompt = (
        "Review the ontology structure (in JSON) below. Identify any relationship that seems questionable or any obvious missing relationship (one issue per line). "
        "If everything looks plausible and complete, reply with 'OK'.\n\n"
        f"{json.dumps(structure, indent=2)}"
    )
    task = Task(description=critique_prompt, expected_output="Critique lines or 'OK'", agent=critic)
    Crew(agents=[critic], tasks=[task], process=Process.sequential).kickoff()
    return str(task.output).strip()

# Assemble ontology RDF graph from the structure (with optional base ontology)
def assemble_ontology_graph(structure: Dict, base_graph: Optional[Graph] = None) -> (Graph, Namespace):
    G = Graph()
    if base_graph:
        G += base_graph  # include base ontology triples if provided
    # Determine base namespace for new ontology
    base_ns = None
    if base_graph:
        for subj in base_graph.subjects(RDF.type, OWL.Ontology):
            base_ns = Namespace(str(subj) + "_ext#")
            break
    if base_ns is None:
        base_ns = Namespace("http://example.org/hydrogen-ontology#")
    # Bind common prefixes
    G.bind("mat", base_ns)  # material/ontology namespace
    G.bind("owl", OWL); G.bind("rdf", RDF); G.bind("rdfs", RDFS); G.bind("xsd", XSD)
    G.bind("skos", Namespace("http://www.w3.org/2004/02/skos/core#"))
    G.bind("prov", Namespace("http://www.w3.org/ns/prov#"))
    G.bind("qudt", Namespace("http://qudt.org/schema/qudt/"))
    G.bind("quantitykind", Namespace("http://qudt.org/vocab/quantitykind/"))
    G.bind("unit", Namespace("http://qudt.org/vocab/unit/"))
    G.bind("chebi", Namespace("http://purl.obolibrary.org/obo/CHEBI_"))
    # Core classes (ensure existence either via base_graph or define minimally)
    core_classes = {
        "Matter": base_ns.Matter,
        "Property": base_ns.Property,
        "Parameter": base_ns.Parameter,
        "Process": base_ns.Process,
        "Metadata": base_ns.Metadata
    }
    for name, uri in core_classes.items():
        if (uri, RDF.type, OWL.Class) not in G:
            G.add((uri, RDF.type, OWL.Class))
    # If base ontology defines a Material class, link our Matter as subclass of it
    if base_graph and (base_ns.Material, RDF.type, OWL.Class) in base_graph:
        G.add((base_ns.Matter, RDFS.subClassOf, base_ns.Material))
    # Define subclasses for process subcategories if present
    if structure.get("manufacturings"):
        cls = base_ns.Manufacturing
        G.add((cls, RDF.type, OWL.Class))
        G.add((cls, RDFS.subClassOf, base_ns.Process))
    if structure.get("measurements"):
        cls = base_ns.Measurement
        G.add((cls, RDF.type, OWL.Class))
        G.add((cls, RDFS.subClassOf, base_ns.Process))
    # Define other category root classes if needed
    if structure.get("instruments"):
        cls = base_ns.Instrument; G.add((cls, RDF.type, OWL.Class)); G.add((cls, RDFS.subClassOf, base_ns.Matter))
    if structure.get("agents"):
        cls = base_ns.Agent; G.add((cls, RDF.type, OWL.Class)); G.add((cls, RDFS.subClassOf, URIRef("http://www.w3.org/ns/prov#Agent")))
    if structure.get("data"):
        cls = base_ns.Data; G.add((cls, RDF.type, OWL.Class)); G.add((cls, RDFS.subClassOf, URIRef("http://www.w3.org/ns/prov#Entity")))
    if structure.get("identifiers"):
        cls = base_ns.Identifier; G.add((cls, RDF.type, OWL.Class)); G.add((cls, RDFS.subClassOf, base_ns.Metadata))
    if structure.get("names"):
        cls = base_ns.Name; G.add((cls, RDF.type, OWL.Class)); G.add((cls, RDFS.subClassOf, base_ns.Metadata))
    if structure.get("values"):
        cls = base_ns.Value; G.add((cls, RDF.type, OWL.Class)); G.add((cls, RDFS.subClassOf, base_ns.Metadata))
    if structure.get("units"):
        cls = base_ns.Unit; G.add((cls, RDF.type, OWL.Class)); G.add((cls, OWL.equivalentClass, URIRef("http://qudt.org/schema/qudt/Unit")))
    # Add each feature class and subclass relation
    category_parent = {
        "matters": base_ns.Matter,
        "properties": base_ns.Property,
        "parameters": base_ns.Parameter,
        "manufacturings": base_ns.Manufacturing if structure.get("manufacturings") else base_ns.Process,
        "measurements": base_ns.Measurement if structure.get("measurements") else base_ns.Process,
        "instruments": base_ns.Instrument,
        "agents": base_ns.Agent,
        "data": base_ns.Data,
        "identifiers": base_ns.Identifier,
        "names": base_ns.Name,
        "values": base_ns.Value,
        "units": base_ns.Unit,
        "metadata": None
    }
    for cat, parent in category_parent.items():
        for item in structure.get(cat, []):
            cls_uri = base_ns[to_valid_uri(item)]
            G.add((cls_uri, RDF.type, OWL.Class))
            if parent and cat != "metadata":
                G.add((cls_uri, RDFS.subClassOf, parent))
    # Add numeric value literals for Value classes
    numeric_prop = base_ns.numericValue
    for val in structure.get("values", []):
        try:
            num_val = float(val)
        except Exception:
            continue
        if (numeric_prop, RDF.type, None) not in G:
            G.add((numeric_prop, RDF.type, OWL.DatatypeProperty))
            G.add((numeric_prop, RDFS.domain, base_ns.Value))
            G.add((numeric_prop, RDFS.range, XSD.double))
            G.add((numeric_prop, RDFS.label, Literal("numeric value", lang="en")))
        val_uri = base_ns[to_valid_uri(val)]
        G.add((val_uri, numeric_prop, Literal(num_val, datatype=XSD.double)))
    # Ensure all necessary object properties exist
    hasP = base_ns.hasProperty; infl = base_ns.influences; hasName = base_ns.hasName; hasId = base_ns.hasIdentifier
    measures = base_ns.measures; usesInst = base_ns.usesInstrument; hasPart = base_ns.hasPart; isPartOf = base_ns.isPartOf
    outData = base_ns.hasOutputData; inMat = base_ns.hasInputMaterial; outMat = base_ns.hasOutputMaterial; inData = base_ns.hasInputData
    hasParam = base_ns.hasParameter; hasSubProc = base_ns.hasSubProcess; isSubProcOf = base_ns.isSubProcessOf
    wasAssoc = URIRef("http://www.w3.org/ns/prov#wasAssociatedWith")
    val_prop = base_ns.hasValue; unit_prop = base_ns.hasUnit
    def ensure_obj_prop(prop_uri, label):
        if (prop_uri, RDF.type, None) not in G:
            G.add((prop_uri, RDF.type, OWL.ObjectProperty))
            G.add((prop_uri, RDFS.label, Literal(label, lang="en")))
    ensure_obj_prop(hasP, "has property")
    if (hasP, RDFS.domain, None) not in G:
        G.add((hasP, RDFS.domain, base_ns.Matter)); G.add((hasP, RDFS.range, base_ns.Property))
    ensure_obj_prop(infl, "influences")
    ensure_obj_prop(hasName, "has name")
    ensure_obj_prop(hasId, "has identifier")
    ensure_obj_prop(measures, "measures")
    if (measures, RDFS.domain, None) not in G:
        G.add((measures, RDFS.domain, base_ns.Measurement)); G.add((measures, RDFS.range, base_ns.Property))
    ensure_obj_prop(usesInst, "uses instrument")
    ensure_obj_prop(hasPart, "has part")
    ensure_obj_prop(isPartOf, "is part of")
    if (isPartOf, OWL.inverseOf, None) not in G:
        G.add((isPartOf, OWL.inverseOf, hasPart))
    ensure_obj_prop(outData, "has output data")
    ensure_obj_prop(inMat, "has input material")
    ensure_obj_prop(outMat, "has output material")
    ensure_obj_prop(inData, "has input data")
    ensure_obj_prop(hasParam, "has parameter")
    ensure_obj_prop(hasSubProc, "has subprocess")
    ensure_obj_prop(isSubProcOf, "is subprocess of")
    if (isSubProcOf, OWL.inverseOf, None) not in G:
        G.add((isSubProcOf, OWL.inverseOf, hasSubProc))
    ensure_obj_prop(val_prop, "has value")
    ensure_obj_prop(unit_prop, "has unit")
    # Align some relations with PROV-O equivalents
    G.add((inMat, OWL.equivalentProperty, URIRef("http://www.w3.org/ns/prov#used")))
    G.add((inData, OWL.equivalentProperty, URIRef("http://www.w3.org/ns/prov#used")))
    G.add((outMat, OWL.equivalentProperty, URIRef("http://www.w3.org/ns/prov#generated")))
    G.add((outData, OWL.equivalentProperty, URIRef("http://www.w3.org/ns/prov#generated")))
    # Add relationships (class-level associations)
    added_rels = set()
    for rel in structure.get("relationships", []):
        s_name = rel.get("subject"); p = rel.get("predicate"); o_name = rel.get("object")
        if not s_name or not p or not o_name:
            continue
        subj_uri = base_ns[to_valid_uri(s_name)]; obj_uri = base_ns[to_valid_uri(o_name)]
        if subj_uri == obj_uri:
            continue
        if p == "hasProperty": pred_uri = hasP
        elif p == "influences": pred_uri = infl
        elif p == "hasName": pred_uri = hasName
        elif p == "hasIdentifier": pred_uri = hasId
        elif p == "measures": pred_uri = measures
        elif p == "usesInstrument": pred_uri = usesInst
        elif p == "hasPart": pred_uri = hasPart
        elif p == "isPartOf": pred_uri = isPartOf
        elif p == "wasAssociatedWith": pred_uri = wasAssoc
        elif p == "hasOutputData": pred_uri = outData
        elif p == "hasInputMaterial": pred_uri = inMat
        elif p == "hasOutputMaterial": pred_uri = outMat
        elif p == "hasInputData": pred_uri = inData
        elif p == "hasParameter": pred_uri = hasParam
        elif p == "hasSubProcess": pred_uri = hasSubProc
        elif p == "isSubProcessOf": pred_uri = isSubProcOf
        elif p == "hasValue": pred_uri = val_prop
        elif p == "hasUnit": pred_uri = unit_prop
        else:
            pred_uri = None
        if pred_uri and (subj_uri, pred_uri, obj_uri) not in added_rels:
            added_rels.add((subj_uri, pred_uri, obj_uri))
            G.add((subj_uri, pred_uri, obj_uri))
    return G, base_ns

# External ontology alignment agent
def map_to_external(class_list: List[tuple]) -> Dict[str, Dict[str, str]]:
    mapping: Dict[str, Dict[str, str]] = {}
    if not class_list:
        return mapping
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    llm = LLM(model="openai/gpt-4", temperature=0.0, max_tokens=600)
    aligner = Agent(
        role="Ontology Alignment Specialist",
        goal="Find equivalent concepts in standard ontologies for the given classes",
        backstory=(
            "You are an expert in ontology alignment (QUDT for quantities/units, ChEBI for chemicals, EMMO for general concepts, PROV-O for agents/data, etc.). "
            "For each class name and category, find an equivalent term in a well-known ontology if it exists."
        ),
        llm=llm
    )
    classes_str = "\n".join([f"- {name} ({category})" for name, category in class_list])
    align_prompt = (
        "Given the list of ontology classes with their categories, find a matching concept in established ontologies/vocabularies if possible. "
        "Use ontologies such as ChEBI for chemical substances, QUDT (QuantityKind or Unit) for physical quantities and units, EMMO for general concepts, PROV-O for agents and data, etc. "
        "Output a JSON mapping each class name to an object with 'ontology' and 'uri' of the best matching concept. "
        "If no suitable match is found for a class, you may omit it or assign an empty object.\n"
        f"Classes:\n{classes_str}"
    )
    task = Task(description=align_prompt, expected_output='JSON mapping class names to ontology matches', agent=aligner)
    Crew(agents=[aligner], tasks=[task], process=Process.sequential).kickoff()
    raw = str(task.output).strip().strip("`")
    try:
        raw_mapping = json.loads(raw)
    except json.JSONDecodeError:
        try:
            raw_mapping = json.loads(raw.replace("'", "\""))
        except Exception:
            raw_mapping = {}
    final_map: Dict[str, Dict[str, str]] = {}
    if isinstance(raw_mapping, dict):
        for cls, info in raw_mapping.items():
            if isinstance(cls, str) and isinstance(info, dict):
                ont = str(info.get("ontology", "")).strip()
                uri = str(info.get("uri", "")).strip()
                if uri:
                    final_map[cls.strip()] = {"ontology": ont, "uri": uri}
    return final_map

def add_external_mappings(G: Graph, mapping: Dict[str, Dict[str, str]], base_ns: Namespace):
    """Add owl:equivalentClass triples for mapped external concepts."""
    for cls_name, info in mapping.items():
        uri = info.get("uri")
        if not uri:
            continue
        cls_uri = base_ns[to_valid_uri(cls_name)]
        if (cls_uri, RDF.type, OWL.Class) not in G:
            continue
        ext_uri = URIRef(uri)
        ont_name = str(info.get("ontology", "")).lower()
        # Bind prefix for known ontologies
        if "chebi" in ont_name or "chebi" in uri.lower():
            G.bind("chebi", "http://purl.obolibrary.org/obo/CHEBI_")
            if uri.startswith("CHEBI:"):
                ext_uri = URIRef(f"http://purl.obolibrary.org/obo/CHEBI_{uri.split(':')[1]}")
        elif "qudt" in ont_name or "quantitykind" in uri.lower():
            G.bind("quantitykind", "http://qudt.org/vocab/quantitykind/")
        elif "unit" in ont_name or "/unit" in uri.lower():
            G.bind("unit", "http://qudt.org/vocab/unit/")
        elif "emmo" in ont_name:
            G.bind("emmo", "http://emmo.info/emmo#")
        elif "prov" in ont_name:
            G.bind("prov", "http://www.w3.org/ns/prov#")
        # Add equivalentClass triple and a comment annotation
        G.add((cls_uri, OWL.equivalentClass, ext_uri))
        if info.get("ontology"):
            G.add((cls_uri, RDFS.comment, Literal(f"Equivalent to {info['ontology']} term {info['uri']}", lang="en")))

def add_labels_and_definitions(G: Graph, synonyms_map: Dict[str, List[str]], structure: Dict, base_ns: Namespace):
    SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    llm = LLM(model="openai/gpt-4", temperature=0.3, max_tokens=100)
    doc_agent = Agent(
        role="Ontology Documentation Specialist",
        goal="Provide concise definitions for ontology classes",
        backstory="You are a scientist writing brief definitions for terms in a hydrogen technology ontology.",
        llm=llm
    )
    for cat, items in structure.items():
        # Skip 'relationships' since it's a list of dicts, not class names
        if cat == "relationships" or not isinstance(items, list):
            continue
        for name in items:
            if not isinstance(name, str):
                continue
            uri = base_ns[to_valid_uri(name)]
            # Preferred label: split CamelCase to words
            label_text = re.sub(CAMEL_RE, " ", name).strip()
            G.add((uri, SKOS.prefLabel, Literal(label_text, lang="en")))
            # Alternative labels from synonyms_map
            for alt in synonyms_map.get(name, []):
                alt_label = re.sub(CAMEL_RE, " ", alt).strip()
                if alt_label and alt_label.lower() != label_text.lower():
                    G.add((uri, SKOS.altLabel, Literal(alt_label, lang="en")))
            # If definition (comment) is not already provided, ask for one
            if (uri, RDFS.comment, None) not in G:
                prompt = f"Provide a one-sentence definition for '{name}' in the context of hydrogen technology."
                task = Task(description=prompt, expected_output="Brief definition sentence.", agent=doc_agent)
                Crew(agents=[doc_agent], tasks=[task], process=Process.sequential).kickoff()
                definition = str(task.output).strip().strip("`")
                if definition:
                    if definition[-1] not in ".!?":
                        definition += "."
                    G.add((uri, RDFS.comment, Literal(definition, lang="en")))

def apply_versioning(G: Graph, base_graph: Optional[Graph]):
    """Add ontology metadata (versioning, imports, priorVersion)."""
    ont_uri = URIRef("http://example.org/hydrogen-ontology")
    if base_graph:
        # If extending an existing ontology, import it and mark this as extension
        for subj in base_graph.subjects(RDF.type, OWL.Ontology):
            ont_uri = URIRef(str(subj) + "_extended")
            G.add((ont_uri, OWL.imports, subj))
            break
    G.add((ont_uri, RDF.type, OWL.Ontology))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    G.add((ont_uri, OWL.versionInfo, Literal(f"Generated on {timestamp}", datatype=XSD.string)))
    # Link to prior version if exists
    if st.session_state.get("last_ont_uri"):
        G.add((ont_uri, OWL.priorVersion, URIRef(st.session_state["last_ont_uri"])))
    st.session_state["last_ont_uri"] = str(ont_uri)

# Suggest Features button action
if st.button("Suggest Features"):
    if not api_key:
        st.error("Please provide your OpenAI API key.")
    else:
        os.environ["OPENAI_API_KEY"] = api_key.strip()
        with st.spinner("GPT-4 is analyzing the question to suggest relevant features..."):
            suggested_features = suggest_features(question)
        if suggested_features:
            st.session_state["features_text"] = suggested_features
            st.success("Feature suggestions added. You may review or edit them below before generating the ontology.")
        else:
            st.warning("No features were suggested. The question might be unclear or too broad. You can manually enter features below.")

# Feature list input (with any suggestions pre-filled)
features_text = st.text_area("**Feature List** (comma-separated):", 
                             value=st.session_state.get("features_text", ""), 
                             height=100, 
                             help="List of features for the ontology. Use CamelCase terms, and you may include context in parentheses (units, typical values, etc.).")
features = [f.strip() for f in features_text.split(",") if f.strip()]
st.session_state["features_text"] = features_text  # update session with current text

if features and not question.strip():
    st.info("No research question provided. Generating a general ontology from the feature list.")

# Generate Ontology button action
if st.button("Generate Ontology"):
    if not api_key:
        st.error("Please enter your OpenAI API key to generate the ontology.")
        st.stop()
    if not features:
        st.error("Feature list is empty. Please enter at least one feature.")
        st.stop()
    os.environ["OPENAI_API_KEY"] = api_key.strip()
    # Step 1: Parse feature context for values/units
    feature_context_map: Dict[str, str] = {}
    base_features: List[str] = []
    for feat in features:
        if "(" in feat and ")" in feat:
            term = feat.split("(", 1)[0].strip()
            context = feat.split("(", 1)[1].rstrip(")")
            context = context.strip()
            if term:
                base_features.append(term)
                if context:
                    feature_context_map[term] = context
        else:
            base_features.append(feat)
    features = base_features
    # Step 2: Find synonyms and merge equivalent terms
    syn_map = find_synonyms(features)
    if syn_map:
        # Merge synonyms: unify any terms that refer to the same concept
        merged_into: Dict[str, str] = {}
        for canon, alts in list(syn_map.items()):
            if canon not in syn_map:
                continue
            for alt in list(alts):
                if alt in syn_map and alt != canon:
                    # Remove the alt term and merge into canon
                    alt_syns = syn_map.pop(alt)
                    # Add alt's synonyms to canon's list
                    for s in alt_syns:
                        if s not in alts and s != canon:
                            alts.append(s)
                    merged_into[alt] = canon
        features = list(syn_map.keys())
        # Merge contexts for unified terms
        for alt_term, canon_term in merged_into.items():
            if alt_term in feature_context_map:
                ctx = feature_context_map.pop(alt_term)
                if ctx:
                    if canon_term in feature_context_map:
                        feature_context_map[canon_term] += "; " + ctx
                    else:
                        feature_context_map[canon_term] = ctx
        # Display merged synonym info to user
        merge_notes = []
        for canon, alts in syn_map.items():
            if alts:
                merge_notes.append(f"**{canon}** (merged synonyms: {', '.join(alts)})")
            else:
                merge_notes.append(f"**{canon}**")
        st.write("**Synonym Merge:** Merged the following equivalent terms:")
        st.write("; ".join(merge_notes))
        st.info("If a merge is incorrect, please adjust the feature list and regenerate.")
    else:
        syn_map = {}
    # Step 3: Build initial ontology structure from features
    with st.spinner("Classifying features and proposing relationships..."):
        structure = build_ontology_structure(question, features)
    if not structure:
        st.stop()  # error already shown to user
    # Step 4: Incorporate values and units from feature context
    unit_replacements: Dict[str, str] = {}
    if feature_context_map:
        for term, context in feature_context_map.items():
            # Extract numeric values
            values_found = re.findall(r"\d+\.?\d*", context)
            values_found = [v for v in values_found if v != ""]
            # Clean context of noise words for unit extraction
            context_clean = context
            for word in ["range", "Range", "typical", "typically", "Typical", "Typically", "approx", "approximately", "Approx", "Approximately"]:
                context_clean = context_clean.replace(word, "")
            # Extract unit-like tokens (non-numeric, non-space/punctuation sequences)
            unit_tokens = re.findall(r"[^\d\s,;()\-\u2013\u2014]+", context_clean)
            unit_tokens = [u for u in unit_tokens if u and not re.match(r"^[\W_]+$", u)]
            # Remove isolated 'e' if scientific notation present (to avoid interpreting 'e' as unit)
            if 'e' in unit_tokens and re.search(r"\d+e[-+]?\d+", context, flags=re.IGNORECASE):
                unit_tokens = [u for u in unit_tokens if u.lower() != 'e']
            unit_tokens = list(dict.fromkeys(unit_tokens))  # unique order-preserving
            # Add Value and Unit entries and relations
            for val in values_found:
                try:
                    float(val)
                except ValueError:
                    continue
                structure.setdefault("values", [])
                if val not in structure["values"]:
                    structure["values"].append(val)
                rel_val = {"subject": term, "predicate": "hasValue", "object": val}
                if rel_val not in structure.get("relationships", []):
                    structure.setdefault("relationships", []).append(rel_val)
            for u in unit_tokens:
                unit_name = u
                # Map special unit symbols to canonical names
                if u in ["C", "degC", "c", "C", "c"]:
                    unit_name = "DegreeCelsius"; unit_replacements[unit_name] = "C"
                elif u in ["F", "degF", "f", "F", "f"]:
                    unit_name = "DegreeFahrenheit"; unit_replacements[unit_name] = "F"
                elif u in ["K", "k"]:
                    unit_name = "Kelvin"; unit_replacements[unit_name] = "K"
                elif u.lower() == "atm":
                    unit_name = "Atmosphere"; unit_replacements[unit_name] = "atm"
                else:
                    if u in ["C", "F"]:
                        # Skip ambiguous single letters (likely part of °C/°F already handled)
                        continue
                    unit_name = u
                structure.setdefault("units", [])
                if unit_name not in structure["units"]:
                    structure["units"].append(unit_name)
                rel_unit = {"subject": term, "predicate": "hasUnit", "object": unit_name}
                if rel_unit not in structure.get("relationships", []):
                    structure.setdefault("relationships", []).append(rel_unit)
    # Step 5: Critique the proposed structure using the critic agent
    feedback = critique_structure(structure)
    if feedback and feedback.lower() != "ok":
        md_box("### Critic Feedback", "warning")
        st.write(feedback)
        # Option to apply critic suggestions by removing flagged relations
        if st.checkbox("Apply critic suggestions (remove flagged relations)", value=True):
            flagged = []
            for rel in structure.get("relationships", []):
                rel_str = f"{rel.get('subject')} {rel.get('predicate')} {rel.get('object')}"
                for line in feedback.splitlines():
                    if rel_str in line:
                        flagged.append(rel)
                        break
            if flagged:
                structure["relationships"] = [r for r in structure["relationships"] if r not in flagged]
                md_box(f"Removed {len(flagged)} relation(s) as suggested by critic.", "info")
    else:
        feedback = "OK"
    # Save results to session for next step
    st.session_state["structure"] = structure
    st.session_state["synonyms_map"] = syn_map
    st.session_state["unit_replacements"] = unit_replacements
    st.session_state["ontology_generated"] = True
    st.success("Ontology structure generated. Review the relationships below before finalizing.")

# Relationship review and confirmation
if st.session_state.get("ontology_generated") and not st.session_state.get("relation_review_done"):
    st.subheader("Review and Edit Proposed Relationships")
    rel_list = []
    for rel in st.session_state["structure"].get("relationships", []):
        s = rel.get("subject"); p = rel.get("predicate"); o = rel.get("object")
        if s and p and o:
            rel_list.append(f"{s} {p} {o}")
    relations_text = st.text_area("Edit relationships (one per line, format: Subject predicate Object):", 
                                  value="\n".join(rel_list), height=150)
    st.write("_You may add, modify, or remove relationships. Use only the allowed predicates listed above._")
    if st.button("Confirm Relationships"):
        final_rels = []
        for line in relations_text.splitlines():
            parts = line.strip().split()
            if len(parts) == 3:
                subj, pred, obj = parts
                if pred in {"hasProperty", "influences", "measures", "hasName", "hasIdentifier", "hasPart", "isPartOf", "wasAssociatedWith", "hasOutputData", "usesInstrument", "hasInputMaterial", "hasOutputMaterial", "hasInputData", "hasParameter", "hasSubProcess", "isSubProcessOf", "hasValue", "hasUnit"}:
                    final_rels.append({"subject": subj, "predicate": pred, "object": obj})
        # Update structure relationships to the confirmed list
        structure = st.session_state["structure"]
        structure["relationships"] = final_rels
        # Identify any new terms introduced in the edited relations
        existing_terms = sum([structure.get(cat, []) for cat in structure if isinstance(structure.get(cat), list)], [])
        introduced = {rel["subject"] for rel in final_rels if rel["subject"] not in existing_terms}
        introduced |= {rel["object"] for rel in final_rels if rel["object"] not in existing_terms}
        # Prepare lists for new terms by category
        new_idents = []; new_names = []
        new_measures = []; new_instruments = []
        new_agents = []; new_data = []
        new_matters = []; new_params = []
        new_values = []; new_units = []
        new_properties = []
        for rel in final_rels:
            s = rel["subject"]; p = rel["predicate"]; o = rel["object"]
            if p == "hasIdentifier" and o in introduced:
                new_idents.append(o); introduced.discard(o)
            if p == "hasName" and o in introduced:
                new_names.append(o); introduced.discard(o)
            if p == "measures":
                if s in introduced:
                    new_measures.append(s); introduced.discard(s)
                if o in introduced:
                    # object is likely a Property
                    new_matters = [x for x in new_matters if x != o]  # remove if wrongly tagged as matter
                    if o not in new_measures and o not in new_properties:
                        new_properties.append(o)
                    introduced.discard(o)
            if p in {"hasPart", "isPartOf"}:
                if s in introduced:
                    new_matters.append(s); introduced.discard(s)
                if o in introduced:
                    new_matters.append(o); introduced.discard(o)
            if p == "usesInstrument":
                if s in introduced:
                    new_measures.append(s); introduced.discard(s)
                if o in introduced:
                    new_instruments.append(o); introduced.discard(o)
            if p == "wasAssociatedWith":
                if s in introduced:
                    new_measures.append(s); introduced.discard(s)
                if o in introduced:
                    new_agents.append(o); introduced.discard(o)
            if p == "hasOutputData":
                if s in introduced:
                    new_measures.append(s); introduced.discard(s)
                if o in introduced:
                    new_data.append(o); introduced.discard(o)
            if p == "hasInputMaterial":
                if s in introduced:
                    new_measures.append(s); introduced.discard(s)
                if o in introduced:
                    new_matters.append(o); introduced.discard(o)
            if p == "hasOutputMaterial":
                if s in introduced:
                    new_measures.append(s); introduced.discard(s)
                if o in introduced:
                    new_matters.append(o); introduced.discard(o)
            if p == "hasInputData":
                if s in introduced:
                    new_measures.append(s); introduced.discard(s)
                if o in introduced:
                    new_data.append(o); introduced.discard(o)
            if p == "hasParameter":
                if s in introduced:
                    new_measures.append(s); introduced.discard(s)
                if o in introduced:
                    new_params.append(o); introduced.discard(o)
            if p in {"hasSubProcess", "isSubProcessOf"}:
                if s in introduced:
                    new_measures.append(s); introduced.discard(s)
                if o in introduced:
                    new_measures.append(o); introduced.discard(o)
            if p == "hasValue":
                if s in introduced:
                    new_params.append(s); introduced.discard(s)
                if o in introduced:
                    new_values.append(o); introduced.discard(o)
            if p == "hasUnit":
                if s in introduced:
                    new_params.append(s); introduced.discard(s)
                if o in introduced:
                    new_units.append(o); introduced.discard(o)
        # Add new terms to their respective categories
        if new_idents:
            structure.setdefault("identifiers", []).extend([x for x in new_idents if x not in structure.get("identifiers", [])])
        if new_names:
            structure.setdefault("names", []).extend([x for x in new_names if x not in structure.get("names", [])])
        if new_measures:
            structure.setdefault("measurements", []).extend([x for x in new_measures if x not in structure.get("measurements", [])])
        if new_instruments:
            structure.setdefault("instruments", []).extend([x for x in new_instruments if x not in structure.get("instruments", [])])
        if new_agents:
            structure.setdefault("agents", []).extend([x for x in new_agents if x not in structure.get("agents", [])])
        if new_data:
            structure.setdefault("data", []).extend([x for x in new_data if x not in structure.get("data", [])])
        if new_matters:
            structure.setdefault("matters", []).extend([x for x in new_matters if x not in structure.get("matters", [])])
        if new_params:
            structure.setdefault("parameters", []).extend([x for x in new_params if x not in structure.get("parameters", [])])
        if new_values:
            structure.setdefault("values", []).extend([x for x in new_values if x not in structure.get("values", [])])
        if new_units:
            structure.setdefault("units", []).extend([x for x in new_units if x not in structure.get("units", [])])
        if new_properties:
            structure.setdefault("properties", []).extend([x for x in new_properties if x not in structure.get("properties", [])])
        if introduced:
            structure.setdefault("metadata", []).extend([x for x in introduced if x not in structure.get("metadata", [])])
        # Deduplicate all category lists
        for cat in ["matters","properties","parameters","manufacturings","measurements","instruments","agents","data","metadata","identifiers","names","values","units"]:
            if cat in structure and isinstance(structure[cat], list):
                structure[cat] = list(dict.fromkeys(structure[cat]))
        st.session_state["structure"] = structure
        st.session_state["relation_review_done"] = True
        st.success("Relationships confirmed. Finalizing ontology...")

# Finalize ontology generation (assemble graph, align, output)
if st.session_state.get("ontology_generated") and st.session_state.get("relation_review_done"):
    structure = st.session_state["structure"]
    syn_map = st.session_state.get("synonyms_map", {})
    unit_replacements = st.session_state.get("unit_replacements", {})
    # Step 6: Adjust categories for consistent domain/range (if needed)
    adjustments = []
    def add_to_category(cat_key: str, term: str):
        structure.setdefault(cat_key, [])
        if term not in structure[cat_key]:
            structure[cat_key].append(term)
            adjustments.append(f"{term} ? {cat_key[:-1].capitalize() if cat_key.endswith('s') else cat_key.capitalize()}")
    for rel in structure.get("relationships", []):
        s = rel.get("subject"); p = rel.get("predicate"); o = rel.get("object")
        if not s or not p or not o:
            continue
        if p == "influences":
            if s not in structure.get("parameters", []) and s not in structure.get("manufacturings", []) and s not in structure.get("measurements", []):
                add_to_category("parameters", s)
            if o not in structure.get("properties", []):
                add_to_category("properties", o)
        elif p == "hasProperty":
            if s not in structure.get("matters", []):
                add_to_category("matters", s)
            if o not in structure.get("properties", []):
                add_to_category("properties", o)
        elif p == "measures":
            if s not in structure.get("measurements", []) and s not in structure.get("manufacturings", []):
                add_to_category("measurements", s)
            if o not in structure.get("properties", []):
                add_to_category("properties", o)
        elif p == "usesInstrument":
            if s not in structure.get("measurements", []) and s not in structure.get("manufacturings", []):
                add_to_category("measurements", s)
            if o not in structure.get("instruments", []):
                add_to_category("instruments", o)
        elif p in {"hasPart", "isPartOf"}:
            if s not in structure.get("matters", []) and s not in structure.get("manufacturings", []) and s not in structure.get("measurements", []):
                add_to_category("matters", s)
            if o not in structure.get("matters", []) and o not in structure.get("manufacturings", []) and o not in structure.get("measurements", []):
                add_to_category("matters", o)
        elif p == "wasAssociatedWith":
            if s not in structure.get("measurements", []) and s not in structure.get("manufacturings", []):
                add_to_category("measurements", s)
            if o not in structure.get("agents", []):
                add_to_category("agents", o)
        elif p == "hasOutputData":
            if s not in structure.get("measurements", []) and s not in structure.get("manufacturings", []):
                add_to_category("measurements", s)
            if o not in structure.get("data", []):
                add_to_category("data", o)
        elif p == "hasInputMaterial":
            if s not in structure.get("measurements", []) and s not in structure.get("manufacturings", []):
                add_to_category("manufacturings" if structure.get("manufacturings") else "measurements", s)
            if o not in structure.get("matters", []):
                add_to_category("matters", o)
        elif p == "hasOutputMaterial":
            if s not in structure.get("measurements", []) and s not in structure.get("manufacturings", []):
                add_to_category("manufacturings" if structure.get("manufacturings") else "measurements", s)
            if o not in structure.get("matters", []):
                add_to_category("matters", o)
        elif p == "hasInputData":
            if s not in structure.get("measurements", []) and s not in structure.get("manufacturings", []):
                add_to_category("measurements", s)
            if o not in structure.get("data", []):
                add_to_category("data", o)
        elif p == "hasParameter":
            if s not in structure.get("measurements", []) and s not in structure.get("manufacturings", []):
                add_to_category("measurements", s)
            if o not in structure.get("parameters", []):
                add_to_category("parameters", o)
        elif p in {"hasSubProcess", "isSubProcessOf"}:
            if s not in structure.get("manufacturings", []) and s not in structure.get("measurements", []):
                add_to_category("measurements", s)
            if o not in structure.get("manufacturings", []) and o not in structure.get("measurements", []):
                add_to_category("measurements", o)
        elif p == "hasValue":
            if s not in structure.get("parameters", []) and s not in structure.get("properties", []):
                # Determine if s is a Property (appears as object of hasProperty/influences) or otherwise a Parameter
                is_prop = any(r.get("object") == s and r.get("predicate") in {"hasProperty", "influences", "measures"} for r in structure.get("relationships", []))
                if is_prop:
                    add_to_category("properties", s)
                else:
                    add_to_category("parameters", s)
            if o not in structure.get("values", []):
                add_to_category("values", o)
        elif p == "hasUnit":
            if s not in structure.get("parameters", []) and s not in structure.get("properties", []):
                is_prop = any(r.get("object") == s and r.get("predicate") in {"hasProperty", "influences", "measures"} for r in structure.get("relationships", []))
                if is_prop:
                    add_to_category("properties", s)
                else:
                    add_to_category("parameters", s)
            if o not in structure.get("units", []):
                add_to_category("units", o)
    if adjustments:
        st.info("**Category Adjustments:** " + "; ".join(adjustments))
    # Deduplicate category lists again
    for cat in ["matters","properties","parameters","manufacturings","measurements","instruments","agents","data","metadata","identifiers","names","values","units"]:
        if cat in structure and isinstance(structure[cat], list):
            structure[cat] = list(dict.fromkeys(structure[cat]))
    # Step 7: Load base ontology (EMMO/MatGraph core) if available
    base_graph = None
    base_urls = [
        "https://raw.githubusercontent.com/MaxDreger92/MatGraph/enhancement/publication/Ontology/MatGraphOntology.ttl",
        "https://raw.githubusercontent.com/MaxDreger92/MatGraph/master/Ontology/MatGraphOntology.ttl"
    ]
    with st.spinner("Loading base ontology (core classes)..."):
        for url in base_urls:
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
            st.warning("Base ontology could not be retrieved; using minimal core classes.")
    # Step 8: Assemble the ontology graph (RDF)
    with st.spinner("Assembling ontology graph..."):
        G, base_ns = assemble_ontology_graph(structure, base_graph)
    # Step 9: Align classes with external ontologies
    class_cat_pairs = []
    for cat_key in ["matters","properties","parameters","manufacturings","measurements","instruments","agents","data","identifiers","names","values","units"]:
        for name in structure.get(cat_key, []):
            cat_label = cat_key[:-1] if cat_key.endswith('s') else cat_key
            if cat_label == "manufacturing": cat_label = "Manufacturing process"
            if cat_label == "measurement": cat_label = "Measurement process"
            class_cat_pairs.append((name, cat_label.capitalize()))
    align_map = {}
    if class_cat_pairs:
        with st.spinner("Aligning classes with external ontologies..."):
            align_map = map_to_external(class_cat_pairs)
            add_external_mappings(G, align_map, base_ns)
    if align_map:
        align_info = [f"{cls} - *{info.get('ontology')}* (`{info.get('uri')}`)" for cls, info in align_map.items()]
        st.write("**External Alignments:** " + "; ".join(align_info))
    else:
        st.write("**External Alignments:** No direct matches found.")
    # Step 10: Add labels (prefLabel/altLabel) and definitions
    # Incorporate unit symbol altLabels using unit_replacements
    for canon, symbol in unit_replacements.items():
        syn_map.setdefault(canon, [])
        if symbol not in syn_map[canon]:
            syn_map[canon].append(symbol)
    with st.spinner("Adding labels and definitions..."):
        add_labels_and_definitions(G, syn_map, structure, base_ns)
    # Step 11: Apply versioning metadata
    apply_versioning(G, base_graph)
    # Step 12: Serialize ontology and provide downloads
    try:
        ttl_data = G.serialize(format="turtle")
    except Exception as e:
        st.error(f"Error serializing ontology to Turtle: {e}")
        ttl_data = None
    if ttl_data:
        st.download_button("Download Ontology (Turtle)", ttl_data, file_name="hydrogen_ontology.ttl", mime="text/turtle")
    # Provide GraphML download of ontology graph
    nxG = nx.MultiDiGraph()
    base_nodes = ["Matter", "Property", "Parameter", "Process"]
    if structure.get("manufacturings"): base_nodes.append("Manufacturing")
    if structure.get("measurements"): base_nodes.append("Measurement")
    if structure.get("instruments"): base_nodes.append("Instrument")
    if structure.get("agents"): base_nodes.append("Agent")
    if structure.get("data"): base_nodes.append("Data")
    if structure.get("metadata"): base_nodes.append("Metadata")
    if structure.get("identifiers"): base_nodes.append("Identifier")
    if structure.get("names"): base_nodes.append("Name")
    if structure.get("values"): base_nodes.append("Value")
    if structure.get("units"): base_nodes.append("Unit")
    for bn in base_nodes:
        nxG.add_node(bn, category=bn, is_base=True)
    # Add class nodes and subclass edges
    for key, parent in [
        ("matters", "Matter"),
        ("properties", "Property"),
        ("parameters", "Parameter"),
        ("manufacturings", "Manufacturing" if structure.get("manufacturings") else "Process"),
        ("measurements", "Measurement" if structure.get("measurements") else "Process"),
        ("instruments", "Instrument"),
        ("agents", "Agent"),
        ("data", "Data"),
        ("identifiers", "Identifier"),
        ("names", "Name"),
        ("values", "Value"),
        ("units", "Unit"),
        ("metadata", "Metadata")
    ]:
        for item in structure.get(key, []):
            node = to_valid_uri(item)
            nxG.add_node(node, category=parent, is_base=False)
            if parent and parent != "Metadata":
                nxG.add_edge(node, parent, label="isA")
    # Add edges for relationships
    for rel in structure.get("relationships", []):
        subj_node = to_valid_uri(rel["subject"]); pred = rel["predicate"]; obj_node = to_valid_uri(rel["object"])
        nxG.add_node(subj_node, category=nxG.nodes[subj_node]["category"] if subj_node in nxG.nodes else "", is_base=False)
        nxG.add_node(obj_node, category=nxG.nodes[obj_node]["category"] if obj_node in nxG.nodes else "", is_base=False)
        nxG.add_edge(subj_node, obj_node, label=pred)
    try:
        graphml_data = nx.generate_graphml(nxG)
        graphml_str = "\n".join(list(graphml_data))
        st.download_button("Download Graph (GraphML)", graphml_str, file_name="hydrogen_ontology.graphml", mime="application/xml")
    except Exception as e:
        st.error(f"Error generating GraphML: {e}")
    # Graph visualization using Graphviz
    try:
        st.subheader("Ontology Graph Visualization")
        dot = Digraph(engine="dot", graph_attr={"rankdir": "TB"})
        shape_map = {
            "Matter": "box", "Property": "ellipse", "Parameter": "diamond", 
            "Process": "polygon", "Manufacturing": "polygon", "Measurement": "polygon",
            "Instrument": "parallelogram", "Agent": "hexagon", "Data": "cylinder",
            "Metadata": "note", "Identifier": "note", "Name": "note",
            "Value": "note", "Unit": "note", "": "oval"
        }
        color_map = {
            "Matter": "#ffdede", "Property": "#deffde", "Parameter": "#dedeff",
            "Process": "#fff2cc", "Manufacturing": "#fff2cc", "Measurement": "#fff2cc",
            "Instrument": "#ffdcb3", "Agent": "#e0ccff", "Data": "#ccffeb",
            "Metadata": "#f0f0f0", "Identifier": "#e0ffff", "Name": "#e0e0ff",
            "Value": "#fff0e0", "Unit": "#ffe0f0", "": "#ffffff"
        }
        for node, data in nxG.nodes(data=True):
            cat = data.get("category", "")
            is_base = data.get("is_base", False)
            shape = shape_map.get(cat, "oval")
            style = "filled" if is_base else "solid"
            fillcolor = color_map.get(cat, "#ffffff") if is_base else "#ffffff"
            label = node if is_base else re.sub(CAMEL_RE, " ", node).strip()
            dot.node(node, label, shape=shape, style=style, fillcolor=fillcolor)
        for u, v, edata in nxG.edges(data=True):
            edge_label = edata.get("label", "")
            dot.edge(u, v, label=edge_label)
        st.graphviz_chart(dot.source)
    except Exception as e:
        st.error(f"Graph visualization failed: {e}")
