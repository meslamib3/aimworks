# -*- coding: utf-8 -*-
# ------------------------------ hydrogen_ontology_generator.py ------------------------------
"""
Hydrogen Technology Ontology Generator (EMMO-based, aligned with RSC semantic model)

Full Streamlit app implementing:
  - Multi-agent pipeline (GPT-4 via CrewAI) for feature suggestion, ontology structuring, critique, etc.
  - Extended semantic categories (Matter, Property, Parameter, Measurement, Manufacturing, Instrument, Agent, Data, Metadata, Identifier, Name, Value, Unit, plus domain-specific ChemicalStressor, RadicalSpecies, DegradationObservable, TestProtocol, ReactionPathway)
  - Additional relations (hasName, hasIdentifier, measures, hasPart, isPartOf, wasAssociatedWith, hasOutputData, usesInstrument, hasInputMaterial, hasOutputMaterial, hasInputData, hasParameter, hasSubProcess, isSubProcessOf, hasValue, hasUnit) 
  - Integration of external ontologies (QUDT, ChEBI, EMMO, PROV-O) for alignment where possible, including numeric QuantityValue pattern for measurable quantities
  - Automated SHACL constraint generation and validation to ensure consistency (with optional auto-fix of violations)
  - Provenance pattern linking Measurements with Instruments and Agents via PROV-O 
  - Option for user to review and edit relationships before finalizing
  - Output ontology in Turtle (with SHACL shapes) and GraphML, plus Graphviz visualization
"""
import streamlit as st
import os, re, json, datetime, textwrap, subprocess, urllib.parse, ast, time
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
try:
    from pyshacl import validate
except ImportError:
    subprocess.run(["pip", "install", "pyshacl"], check=True)
    from pyshacl import validate

# Streamlit page configuration
st.set_page_config(page_title="Hydrogen Technology Ontology Generator (EMMO-based)", page_icon="??", layout="centered")
st.title("Hydrogen Technology Ontology Generator (EMMO-based)")

st.write(
    "**Description:** Provide a hydrogen technology research question (and optionally a list of features) to automatically generate an extended ontology. "
    "Multiple GPT-4 agents (feature suggestion, terminologist, domain ontologist, coverage checker, etc.) collaborate to identify relevant concepts (categorized as Matter, Property, Parameter, Manufacturing, Measurement, Instrument, Agent, Data, Metadata, Identifier, Name, Value, Unit, as well as domain-specific ChemicalStressor, RadicalSpecies, DegradationObservable, TestProtocol, ReactionPathway) and link them with meaningful relationships (hasProperty, influences, measures, hasName, hasIdentifier, hasPart, isPartOf, wasAssociatedWith, hasOutputData, usesInstrument, hasInputMaterial, hasOutputMaterial, hasInputData, hasParameter, hasSubProcess, isSubProcessOf, hasValue, hasUnit). "
    "The app provides an interactive scenario builder to draft complete multi-step research workflows covering all relevant aspects. The ontology is then constructed from the selected scenario or provided features, and formal consistency is enforced through SHACL validation (with an option to auto-fix issues). "
    "A review step allows you to confirm or edit the proposed relationships before finalizing. "
    "The final ontology (including constraints) can be downloaded in Turtle or JSON-LD or GraphML format, and a graph visualization is provided for inspection."
)

# User Inputs
question: str = st.text_input("**Research Question:**", 
    help="E.g. 'How does operating temperature affect hydrogen permeability of a Nafion membrane in PEM fuel cells?'")
api_key: str = st.text_input("**OpenAI API Key** (required for GPT-4 access):", type="password")
mode = st.radio("**Generation Mode:**", ["Quick (fewer LLM calls)", "Detailed (comprehensive)"], index=1, 
               help="Quick mode skips some steps (synonyms, detailed definitions, full validation) for faster results. Detailed mode runs all steps.")
quick_mode = mode.startswith("Quick")

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
if "scenarios" not in st.session_state:
    st.session_state["scenarios"] = []
if "scenario_selected" not in st.session_state:
    st.session_state["scenario_selected"] = None

# Utility functions
CAMEL_RE = re.compile(r'(?<=[a-z])(?=[A-Z])')

# --- scenario ? bullet parser ------------------------------------
import re
_HEADING_RE = re.compile(
    r"\*\*(Matter|Property|Parameter|Manufacturing|Measurement|Instrument|"
    r"Agent|Data|Metadata|Identifier|Name|Value|Unit)\*\s*:\s*", re.I)

def _parse_scenario_bullets(txt: str) -> dict:
    """returns {heading:[items,â€¦]} extracted from plain-text scenario"""
    parts = _HEADING_RE.split(txt)
    buckets = {}
    for i in range(1, len(parts), 2):
        head = parts[i].title()
        first_line = parts[i + 1].split('\n', 1)[0]
        buckets[head] = [t.strip() for t in first_line.split(',') if t.strip()]
    return buckets

def features_from_scenario_text(txt: str) -> list[str]:
    """flat list of all items that appear under the 13 bold headings"""
    flat = []
    for items in _parse_scenario_bullets(txt).values():
        for tok in items:
            if tok not in flat:
                flat.append(tok)
    return flat


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

def guess_category(term: str) -> Optional[str]:
    """Heuristically guess category of a term based on keywords."""
    low = term.lower()
    if any(x in low for x in ["test", "experiment", "protocol", "process"]):
        return "measurements"
    if "radical" in low:
        return "radicalSpecies"
    if any(x in low for x in ["peroxide", "acid"]):
        return "chemicalStressors"
    if any(x in low for x in ["rate", "slope", "coefficient", "constant"]):
        return "properties"
    return None



# ------------------------------------------------------------------
# Scenario-Builder (narrative version) – single, comprehensive scenario
# ------------------------------------------------------------------
def propose_scenarios(question_text: str) -> List[str]:
    """
    Expand a user-supplied research question into ONE industrial-grade,
    Nature-style abstract that *explicitly* covers every ontology bucket
    and enumerates every required predicate once.

    Returns
    -------
    List[str]
        A list containing one polished scenario (len == 1).

    Scenario structure
    ------------------
    • Title – = 100 chars  
    • ¶1  – Rationale & scope (~250–300 words)  
    • 14 bold headings in the *exact* order below
        Matter / Property / Parameter / Manufacturing / Measurement /
        Instrument / Agent / Data / Metadata / Identifier / Name /
        Value / Unit / Relationships

      – The first 13 headings list CamelCase items (context in (…) allowed).  
      – The **Relationships** heading must contain **18 triples** –
        one line for each predicate in the set shown in the prompt.

    • Concluding sentences may follow, but every bullet item and every
      predicate must be referenced in the narrative.  Cite = 1 ISO / IEC / ASTM
      standard.
    """
    import re, streamlit as st

    # -- Guards ----------------------------------------------------
    if not question_text.strip():
        st.warning("Please enter a research question before generating the scenario.")
        return []
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OpenAI API key is missing – cannot contact GPT-4.")
        return []

    openai.api_key = os.getenv("OPENAI_API_KEY")

    # -- LLM & agents ---------------------------------------------
    author_llm  = LLM(model="openai/gpt-4", temperature=0.25, max_tokens=2400)
    critic_llm  = LLM(model="openai/gpt-4", temperature=0.00, max_tokens=600)
    repair_llm  = LLM(model="openai/gpt-4", temperature=0.00, max_tokens=2400)

    author = Agent(
        role="Hydrogen-Tech Scenario Author",
        goal=("Draft a single, detailed, industrial-grade hydrogen-energy "
              "research scenario with full ontology coverage."),
        backstory=("You devise TRL-9 R&D programmes for PEM fuel-cell and "
                   "electrolyser manufacturers; expert in materials science, "
                   "process engineering, digital twins and standards."),
        llm=author_llm,
    )

    critic = Agent(
        role="Scenario Critic",
        goal=("Ensure the scenario is complete, quantitative and references "
              "every ontology bucket and every required predicate."),
        backstory=("You audit hydrogen-energy research plans; you reject "
                   "vagueness, missing units or absent ontology headings."),
        llm=critic_llm,
    )

    repairer = Agent(
        role="Scenario Repairer",
        goal="Fix every deficiency flagged by the critic without changing style.",
        backstory="You polish research abstracts until they pass strict audits.",
        llm=repair_llm,
    )

    # -- Example stub (kept short – just for style) ---------------
    example_stub = (
        "Impact of Ionomer-to-Carbon Ratio on Polarization Curve Performance in PEM Fuel Cells\n\n"
        "**Matter:** NafionD2020, VulcanXC72\n"
        "**Property:** PolarizationCurve, IonicConductivity (S/cm)\n"
        "**Parameter:** IonomerToCarbonRatio (0.4–1.2 —), PtLoading (0.20 mg Pt/cm²)\n"
        "**Manufacturing:** UltrasonicInkDispersion (20 kHz, 60 min)\n"
        "**Measurement:** PolarizationSweep (0.9–0.3 V), ElectrochemicalImpedance (0.1 Hz–10 kHz)\n"
        "**Instrument:** FuelConC1000LT, BiologicSP300\n"
        "**Agent:** FuelCellScientist, LabTechnician\n"
        "**Data:** RawPolarizationCurves, ImpedanceSpectra\n"
        "**Metadata:** LabTemperature (22 °C), RelativeHumidity (50 %)\n"
        "**Identifier:** MEABatchID\n"
        "**Name:** RunCodeICRatio240601\n"
        "**Value:** IonomerToCarbonRatio 0.4, 0.6, 0.8, 1.0, 1.2\n"
        "**Unit:** DegreeCelsius, Percent, Volt, Hz, S/cm\n"
        "**Relationships:**\n"
        "NafionD2020  hasProperty  IonicConductivity\n"
        "IonomerToCarbonRatio  influences  IonicConductivity\n"
        "ElectrochemicalImpedance  measures  IonicConductivity\n"
        "NafionD2020  hasName  RunCodeICRatio240601\n"
        "NafionD2020  hasIdentifier  MEABatchID\n"
        "MembraneAssembly  hasPart  NafionD2020\n"
        "NafionD2020  isPartOf  MembraneAssembly\n"
        "ElectrochemicalImpedance  wasAssociatedWith  FuelCellScientist\n"
        "ElectrochemicalImpedance  hasOutputData  ImpedanceSpectra\n"
        "ElectrochemicalImpedance  usesInstrument  BiologicSP300\n"
        "ElectrochemicalImpedance  hasInputMaterial  NafionD2020\n"
        "UltrasonicInkDispersion  hasOutputMaterial  CatalystInk\n"
        "ModelValidation  hasInputData  RawPolarizationCurves\n"
        "ModelValidation  hasParameter  IonomerToCarbonRatio\n"
        "ModelValidation  hasSubProcess  PolarizationSweep\n"
        "PolarizationSweep  isSubProcessOf  ModelValidation\n"
        "IonomerToCarbonRatio  hasValue  0.8\n"
        "IonomerToCarbonRatio  hasUnit  Dimensionless\n"
    )

    # -- Prompts --------------------------------------------------
    author_prompt = f"""
**TASK – AUTHOR**

Write ONE detailed, hypothetical research scenario answering:

“{question_text}”

Follow the style indicated in the stub below **but make the content unique**:

{example_stub}

**Formatting rules**

1. Intro paragraph: 250–300 words.  
2. Then list **14 bold headings in exactly this order**  
   Matter / Property / Parameter / Manufacturing / Measurement / Instrument /
   Agent / Data / Metadata / Identifier / Name / Value / Unit / Relationships
3.  Items under the first 13 headings **must be CamelCase**; you may append
    context in parentheses (units, ranges, methods, etc.).  
4.  Under **Relationships** write *exactly 18 lines*, one for each predicate:

    hasProperty, influences, measures, hasName, hasIdentifier, hasPart,
    isPartOf, wasAssociatedWith, hasOutputData, usesInstrument,
    hasInputMaterial, hasOutputMaterial, hasInputData, hasParameter,
    hasSubProcess, isSubProcessOf, hasValue, hasUnit

    Use the syntax  
    `Subject  predicate  Object`  ? two spaces around the predicate.
5.  Every bullet item and every predicate must be referenced at least once in
    the narrative.  
6.  Cite = 1 recognised standard (ISO / IEC / ASTM).  
7.  Return **only** the scenario text – no back-ticks, no JSON, no commentary.
"""

    critic_prompt = """
**TASK – CRITIC**

Reject unless ALL conditions hold:

? 14 bold headings present and in correct order.  
? First 13 headings list = 1 CamelCase item each (context in (…) allowed).  
? Every Parameter / Value line has a numeric figure **and** unit.  
? Under **Relationships** there are **exactly 18 triples**, each using one of
   the required predicates **once and only once**, formatted
   `Subject  predicate  Object`.  
? Narrative references every bullet item and every predicate.  
? Narrative length 200–350 words and cites = 1 ISO / IEC / ASTM standard.

If everything passes, reply **OK**. Otherwise list problems as bullet points.
"""

    # -- Run author ? critic --------------------------------------
    author_task = Task(description=author_prompt,
                       expected_output="Scenario text",
                       agent=author)

    critic_task = Task(description=critic_prompt,
                       expected_output="'OK' or list of issues",
                       agent=critic,
                       context=[author_task])

    Crew(agents=[author, critic],
         tasks=[author_task, critic_task],
         process=Process.sequential).kickoff()

    scenario_text  = str(author_task.output).strip()
    critique_text  = str(critic_task.output).strip()

    # -- One repair pass if critic not happy ----------------------
    if critique_text.lower() != "ok":
        repair_prompt = (
            "The critic found these issues:\n"
            f"{critique_text}\n\n"
            "Rewrite the scenario so all issues are resolved. "
            "Keep the same title, bucket order and overall style.\n\n"
            f"Original scenario:\n{scenario_text}"
        )
        repair_task = Task(description=repair_prompt,
                           expected_output="Corrected scenario",
                           agent=repairer)

        Crew(agents=[repairer],
             tasks=[repair_task],
             process=Process.sequential).kickoff()

        scenario_text = str(repair_task.output).strip()

    # -- Final clean-up -------------------------------------------
    scenario_text = re.sub(r"^```.*?```$",
                           "",
                           scenario_text,
                           flags=re.DOTALL | re.MULTILINE).strip()

    return [scenario_text]


# Feature suggestion agent (Domain Expert)
def suggest_features(question_text: str) -> str:
    if not question_text.strip():
        return ""
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    llm = LLM(model="openai/gpt-4", temperature=0.2, max_tokens=512)
    domain_expert = Agent(
        role="Hydrogen Domain Expert",
        goal=(
            "Identify key materials, properties, parameters, processes, and other relevant features in hydrogen technology related to the given research question. "
            "Include typical units, value ranges, or measurement methods for each feature where applicable."
        ),
        backstory=(
            "You are a senior researcher in hydrogen technology. You know typical experimental conditions (temperatures, pressures, catalysts), materials (electrolytes, catalysts, membranes), performance metrics (efficiencies, yields, degradation rates), and processes (electrolysis, storage, polymer degradation). "
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
    # (Removed degradation-specific prompt injection for simplicity)
    task = Task(description=prompt, expected_output="Comma-separated list of features with context in parentheses", agent=domain_expert)
    Crew(agents=[domain_expert], tasks=[task], process=Process.sequential).kickoff()
    output = str(task.output).strip().strip("`")
    return output.rstrip(", ")

def build_combined_feature_list(question: str, scenario_txt: str) -> str:
    """scenario bullets  +  LLM suggestions  -->  merged comma string"""
    scen_feats = features_from_scenario_text(scenario_txt)
    llm_raw = suggest_features(question)
    llm_feats = [f.strip() for f in llm_raw.split(',') if f.strip()]

    merged = []
    for feat in scen_feats + llm_feats:
        if feat not in merged:
            merged.append(feat)
    return ", ".join(merged)


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

@st.cache_data(show_spinner=False)
def cached_find_synonyms(features_tuple: tuple) -> Dict[str, List[str]]:
    """Cached wrapper for find_synonyms to reduce repeated GPT calls."""
    return find_synonyms(list(features_tuple))

# Ontology structuring agents (Domain Expert + Ontology Engineer)
def build_ontology_structure(question_text: str, feature_list: List[str]) -> Optional[Dict]:
    if not feature_list:
        return None
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    # Agent to analyze and categorize features, propose relationships
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
        "Additionally, if any Parameter or Property feature has a specific numeric value or unit given (e.g. '80 Ãƒâ€šÃ‚Â°C'), treat it as a physical quantity. We will model such features as a QuantityKind with an associated numeric value and unit (as a QuantityValue individual in the graph). "
        "For every Measurement process introduced, also include at least one Instrument (equipment) and one Agent (Person or Organization) involved, and relate them using usesInstrument (Measurement -> Instrument) and wasAssociatedWith (Measurement -> Agent).\n\n"
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
        try:
            # Try using Python's literal eval for lenient parsing (accepts single quotes, trailing commas)
            py_obj = ast.literal_eval(json_str)
            if isinstance(py_obj, dict):
                structure = py_obj
            else:
                raise ValueError("Parsed JSON is not a dict")
        except Exception:
            fix_str = json_str.replace("'", "\"")
            fix_str = re.sub(r",\s*}", "}", fix_str)
            fix_str = re.sub(r",\s*\]", "]", fix_str)
            try:
                structure = json.loads(fix_str)
            except json.JSONDecodeError:
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

# (Degradation-specific helpers removed: ensure_domain_coverage and inject_reaction_patterns are not used)

# Assemble ontology RDF graph from the structure (with optional base ontology)
def assemble_ontology_graph(structure: Dict, base_graph: Optional[Graph] = None) -> (Graph, Namespace):
    G = Graph()
    if base_graph:
        G += base_graph
    # Determine base namespace
    base_ns = None
    if base_graph:
        for subj in base_graph.subjects(RDF.type, OWL.Ontology):
            base_ns = Namespace(str(subj) + "_ext#")
            break
    if base_ns is None:
        base_ns = Namespace("https://w3id.org/h2kg/hydrogen-ontology#")
    # Bind prefixes
    G.bind("mat", base_ns)
    G.bind("owl", OWL); G.bind("rdf", RDF); G.bind("rdfs", RDFS); G.bind("xsd", XSD)
    G.bind("skos", Namespace("http://www.w3.org/2004/02/skos/core#"))
    G.bind("prov", Namespace("http://www.w3.org/ns/prov#"))
    G.bind("qudt", Namespace("http://qudt.org/schema/qudt/"))
    G.bind("quantitykind", Namespace("http://qudt.org/vocab/quantitykind/"))
    G.bind("unit", Namespace("http://qudt.org/vocab/unit/"))
    G.bind("chebi", Namespace("http://purl.obolibrary.org/obo/CHEBI_"))
    # Core classes
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
    # Extend base ontology class hierarchy if base provided
    if base_graph and (base_ns.Material, RDF.type, OWL.Class) in base_graph:
        G.add((base_ns.Matter, RDFS.subClassOf, base_ns.Material))
    # Define process subcategories if present
    if structure.get("manufacturings"):
        cls = base_ns.Manufacturing
        G.add((cls, RDF.type, OWL.Class)); G.add((cls, RDFS.subClassOf, base_ns.Process))
    if structure.get("measurements"):
        cls = base_ns.Measurement
        G.add((cls, RDF.type, OWL.Class)); G.add((cls, RDFS.subClassOf, base_ns.Process))
    # Domain-specific category classes
    if structure.get("chemicalStressors") or structure.get("radicalSpecies"):
        cls = base_ns.ChemicalStressor
        G.add((cls, RDF.type, OWL.Class)); G.add((cls, RDFS.subClassOf, base_ns.Matter))
    if structure.get("radicalSpecies"):
        cls = base_ns.RadicalSpecies
        G.add((cls, RDF.type, OWL.Class)); G.add((cls, RDFS.subClassOf, base_ns.ChemicalStressor))
    if structure.get("degradationObservables"):
        cls = base_ns.DegradationObservable
        G.add((cls, RDF.type, OWL.Class)); G.add((cls, RDFS.subClassOf, base_ns.Property))
    if structure.get("testProtocols"):
        cls = base_ns.TestProtocol
        G.add((cls, RDF.type, OWL.Class))
        if structure.get("measurements") or (base_ns.Measurement, RDF.type, OWL.Class) in G:
            G.add((cls, RDFS.subClassOf, base_ns.Measurement))
        else:
            G.add((cls, RDFS.subClassOf, base_ns.Process))
    if structure.get("reactionPathways"):
        cls = base_ns.ReactionPathway
        G.add((cls, RDF.type, OWL.Class)); G.add((cls, RDFS.subClassOf, base_ns.Process))
    # Other category root classes
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
        "metadata": None,
        "chemicalStressors": base_ns.ChemicalStressor,
        "radicalSpecies": base_ns.RadicalSpecies if structure.get("radicalSpecies") else base_ns.ChemicalStressor,
        "degradationObservables": base_ns.DegradationObservable,
        "testProtocols": base_ns.TestProtocol if structure.get("testProtocols") else (base_ns.Measurement if structure.get("measurements") else base_ns.Process),
        "reactionPathways": base_ns.ReactionPathway
    }
    for cat, parent in category_parent.items():
        for item in structure.get(cat, []):
            cls_uri = base_ns[to_valid_uri(item)]
            G.add((cls_uri, RDF.type, OWL.Class))
            if parent and cat != "metadata":
                G.add((cls_uri, RDFS.subClassOf, parent))
    # Numeric value and unit relations handling (QUDT pattern)
    param_value_map: Dict[str, Dict[str, Optional[str]]] = {}
    cleaned_rels = []
    for rel in structure.get("relationships", []):
        subj = rel.get("subject"); pred = rel.get("predicate"); obj = rel.get("object")
        if not subj or not pred or not obj:
            continue
        if pred == "hasValue":
            param_value_map.setdefault(subj, {})["value"] = str(obj).strip()
            continue
        elif pred == "hasUnit":
            param_value_map.setdefault(subj, {})["unit"] = str(obj).strip()
            continue
        else:
            cleaned_rels.append(rel)
    structure["relationships"] = cleaned_rels
    # Ensure numeric pattern for each collected parameter/property value
    known_unit_uris = {
        "DegreeCelsius": "http://qudt.org/vocab/unit#DegreeCelsius",
        "DegreeFahrenheit": "http://qudt.org/vocab/unit#DegreeFahrenheit",
        "Kelvin": "http://qudt.org/vocab/unit#Kelvin",
        "Atmosphere": "http://qudt.org/vocab/unit#Atmosphere",
        "Bar": "http://qudt.org/vocab/unit#Bar",
        "Pascal": "http://qudt.org/vocab/unit#Pascal",
        "Percent": "http://qudt.org/vocab/unit#Percent",
        "Second": "http://qudt.org/vocab/unit#Second",
        "Minute": "http://qudt.org/vocab/unit#Minute",
        "Hour": "http://qudt.org/vocab/unit#Hour"
    }
    dimensionless_unit = URIRef("http://qudt.org/vocab/unit#Dimensionless")
    for subj_name, vals in param_value_map.items():
        if not vals.get("value"):
            continue
        subj_uri = base_ns[to_valid_uri(subj_name)]
        G.add((subj_uri, RDF.type, URIRef("http://qudt.org/schema/qudt/QuantityKind")))
        try:
            numeric_val = float(vals["value"])
        except Exception:
            try:
                numeric_val = float(vals["value"].replace(",", "").strip())
            except Exception:
                continue
        unit_name = vals.get("unit")
        if unit_name:
            unit_key = to_valid_uri(unit_name)
            if unit_key in known_unit_uris:
                unit_uri = URIRef(known_unit_uris[unit_key])
            else:
                unit_uri = base_ns[to_valid_uri(unit_name)]
        else:
            unit_uri = dimensionless_unit
        qv = BNode()
        G.add((qv, RDF.type, URIRef("http://qudt.org/schema/qudt/QuantityValue")))
        G.add((qv, URIRef("http://qudt.org/schema/qudt/numericValue"), Literal(numeric_val, datatype=XSD.double)))
        G.add((qv, URIRef("http://qudt.org/schema/qudt/unit"), unit_uri))
        G.add((qv, URIRef("http://qudt.org/schema/qudt/quantityKind"), subj_uri))
        G.add((subj_uri, base_ns.hasQuantityValue, qv))
    # Ensure object properties exist with basic domain/range where obvious
    hasP = base_ns.hasProperty; infl = base_ns.influences; hasName = base_ns.hasName; hasId = base_ns.hasIdentifier
    measures = base_ns.measures; usesInst = base_ns.usesInstrument; hasPart = base_ns.hasPart; isPartOf = base_ns.isPartOf
    outData = base_ns.hasOutputData; inMat = base_ns.hasInputMaterial; outMat = base_ns.hasOutputMaterial; inData = base_ns.hasInputData
    hasParam = base_ns.hasParameter; hasSubProc = base_ns.hasSubProcess; isSubProcOf = base_ns.isSubProcessOf
    wasAssoc = URIRef("http://www.w3.org/ns/prov#wasAssociatedWith")
    val_prop = base_ns.hasValue; unit_prop = base_ns.hasUnit; hasQVal = base_ns.hasQuantityValue
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
    if (hasParam, RDFS.domain, None) not in G:
        G.add((hasParam, RDFS.domain, base_ns.Process)); G.add((hasParam, RDFS.range, base_ns.Parameter))
    ensure_obj_prop(hasSubProc, "has subprocess")
    ensure_obj_prop(isSubProcOf, "is subprocess of")
    if (isSubProcOf, OWL.inverseOf, None) not in G:
        G.add((isSubProcOf, OWL.inverseOf, hasSubProc))
    ensure_obj_prop(val_prop, "has value")
    ensure_obj_prop(unit_prop, "has unit")
    ensure_obj_prop(hasQVal, "has quantity value")
    if (hasQVal, RDFS.range, None) not in G:
        G.add((hasQVal, RDFS.range, URIRef("http://qudt.org/schema/qudt/QuantityValue")))
    # Align input/output properties with PROV-O
    G.add((inMat, OWL.equivalentProperty, URIRef("http://www.w3.org/ns/prov#used")))
    G.add((inData, OWL.equivalentProperty, URIRef("http://www.w3.org/ns/prov#used")))
    G.add((outMat, OWL.equivalentProperty, URIRef("http://www.w3.org/ns/prov#generated")))
    G.add((outData, OWL.equivalentProperty, URIRef("http://www.w3.org/ns/prov#generated")))
    # Add relationships (class-to-class links)
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

# External ontology alignment (BioPortal + GPT)
BIOPORTAL_KEY = os.getenv('BIOPORTAL_KEY', '')
def find_external_mapping(label: str) -> Optional[Dict[str, str]]:
    if not BIOPORTAL_KEY:
        return None
    query_text = re.sub(CAMEL_RE, " ", label).strip()
    try:
        res = requests.get("https://data.bioontology.org/search", params={"q": query_text, "apikey": BIOPORTAL_KEY}, timeout=5)
        if res.status_code != 200:
            return None
        data = res.json()
    except Exception:
        return None
    results = data.get("collection") or data.get("results") or []
    best_score = 0.0
    best_match = None
    for item in results:
        score = item.get("score") or item.get("annotationsScore") or 0
        try:
            score = float(score)
        except:
            score = 0
        if score > best_score:
            best_score = score
            best_match = item
        if best_score >= 0.9:
            break
    if best_match and best_score >= 0.9:
        ont = best_match.get("ontologyAcronym") or best_match.get("ontologyName") or best_match.get("links", {}).get("ontology")
        uri = best_match.get("@id") or best_match.get("id")
        if uri:
            return {"ontology": str(ont) if ont else "", "uri": uri}
    return None

def map_to_external(class_list: List[tuple]) -> Dict[str, Dict[str, str]]:
    mapping: Dict[str, Dict[str, str]] = {}
    if not class_list:
        return mapping
    # First attempt mapping via BioPortal for each class
    for name, category in class_list:
        result = find_external_mapping(name)
        if result:
            mapping[name] = {"ontology": result.get("ontology", ""), "uri": result.get("uri", "")}
    # Prepare classes not mapped yet for GPT aligner
    remaining = [(name, cat) for name, cat in class_list if name not in mapping]
    if not remaining:
        return mapping
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    llm = LLM(model="openai/gpt-4", temperature=0.0, max_tokens=600)
    aligner = Agent(
        role="Ontology Alignment Specialist",
        goal="Find equivalent concepts in standard ontologies for the given classes",
        backstory=(
            "You are an expert in ontology alignment (QUDT for quantities/units, ChEBI for chemicals, EMMO for general concepts, PROV-O for agents and data, etc.). "
            "For each class name and category, find a matching term in a well-known ontology if it exists, and provide the ontology name and URI."
        ),
        llm=llm
    )
    classes_str = "\n".join([f"- {name} ({category})" for name, category in remaining])
    align_prompt = (
        "Given the list of ontology classes with their categories, find a matching concept in established ontologies/vocabularies if possible. "
        "Use ontologies such as ChEBI for chemical substances, QUDT (QuantityKind or Unit) for physical quantities and units, EMMO for general concepts, PROV-O for agents and data, etc. "
        "Output a JSON mapping each class name to an object with 'ontology' and 'uri' for the best matching external concept. "
        "If no suitable match is found for a class, you may omit it or use an empty object for that class.\n"
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
    if isinstance(raw_mapping, dict):
        for cls, info in raw_mapping.items():
            if isinstance(cls, str) and isinstance(info, dict):
                ont = str(info.get("ontology", "")).strip()
                uri = str(info.get("uri", "")).strip()
                if uri:
                    mapping[cls.strip()] = {"ontology": ont, "uri": uri}
    return mapping

def add_external_mappings(G: Graph, mapping: Dict[str, Dict[str, str]], base_ns: Namespace):
    for cls_name, info in mapping.items():
        uri = info.get("uri")
        if not uri:
            continue
        cls_uri = base_ns[to_valid_uri(cls_name)]
        if (cls_uri, RDF.type, OWL.Class) not in G:
            continue
        ext_uri = URIRef(uri)
        ont_name = str(info.get("ontology", "")).lower()
        # Bind known ontology prefixes if applicable
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
        # Link class to external concept
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
        if cat == "relationships" or not isinstance(items, list):
            continue
        for name in items:
            if not isinstance(name, str):
                continue
            uri = base_ns[to_valid_uri(name)]
            # Preferred label (SKOS prefLabel)
            label_text = re.sub(CAMEL_RE, " ", name).strip()
            G.add((uri, SKOS.prefLabel, Literal(label_text, lang="en")))
            # Alternative labels from synonyms (SKOS altLabel)
            for alt in synonyms_map.get(name, []):
                alt_label = re.sub(CAMEL_RE, " ", alt).strip()
                if alt_label and alt_label.lower() != label_text.lower():
                    G.add((uri, SKOS.altLabel, Literal(alt_label, lang="en")))
            # Definition (RDFS comment)
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
    ont_uri = URIRef("https://w3id.org/h2kg/hydrogen-ontology")
    if base_graph:
        for subj in base_graph.subjects(RDF.type, OWL.Ontology):
            ont_uri = URIRef(str(subj) + "_extended")
            G.add((ont_uri, OWL.imports, subj))
            break
    G.add((ont_uri, RDF.type, OWL.Ontology))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    G.add((ont_uri, OWL.versionInfo, Literal(f"Generated on {timestamp}", datatype=XSD.string)))
    if st.session_state.get("last_ont_uri"):
        G.add((ont_uri, OWL.priorVersion, URIRef(st.session_state["last_ont_uri"])))
    st.session_state["last_ont_uri"] = str(ont_uri)

# Critic agent to review the drafted ontology structure
def critique_structure(structure: Dict) -> str:
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    llm = LLM(model="openai/gpt-4", temperature=0.0, max_tokens=400)
    critic = Agent(
        role="Ontology Critic",
        goal="Critique the proposed ontology structure for any implausible or missing elements",
        backstory="You are a critical reviewer who checks the draft ontology for incorrect relations or missing important links. You also note if any key class lacks an external mapping or a definition.",
        llm=llm
    )
    critique_prompt = (
        "Review the ontology structure (in JSON) below. Identify any relationship that seems questionable or any obvious missing relationship (one issue per line). "
        "Also flag any class that appears important but lacks a mapping to an external standard ontology or a definition comment. "
        "If everything looks plausible and complete, reply with 'OK'.\n\n"
        f"{json.dumps(structure, indent=2)}"
    )
    task = Task(description=critique_prompt, expected_output="List of issues or 'OK'", agent=critic)
    Crew(agents=[critic], tasks=[task], process=Process.sequential).kickoff()
    return str(task.output).strip()

# Scenario-Builder Stage
if st.button("Generate Scenarios"):
    if not api_key:
        st.error("Please provide your OpenAI API key.")
    else:
        os.environ["OPENAI_API_KEY"] = api_key.strip()
        with st.spinner("GPT-4 is generating scenario drafts..."):
            scenario_list = propose_scenarios(question)
        if scenario_list:
            st.session_state["scenarios"] = scenario_list
            st.session_state["scenario_selected"] = None

# Display scenario proposals if available
if st.session_state.get("scenarios"):
    st.subheader("Scenario Proposals")
    st.write("Three alternative research scenarios have been generated. Expand each panel to review and edit the JSON, then select one to proceed:")
    for idx, scen in enumerate(st.session_state["scenarios"], start=1):
        with st.expander(f"Scenario {idx}", expanded=False):
            edited = st.text_area(
                f"Scenario {idx} text",
                value=scen,                      # plain text, not JSON
                height=300,
                key=f"scen_{idx}"
            )

            if st.button(f"Use scenario {idx} & expand features"):
                scen_txt = edited.strip()
                st.session_state["scenario_selected"] = scen_txt

                # build combined list = scenario bullets + LLM suggestions
                os.environ["OPENAI_API_KEY"] = api_key.strip()
                combined = build_combined_feature_list(question, scen_txt)

                st.session_state["features_text"] = combined
                st.success("Feature list expanded with items mined from the scenario.")

# Button: Suggest Features (alternative path if not using scenarios)
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

# Feature list input
features_text = st.text_area("**Feature List** (comma-separated):", 
                             value=st.session_state.get("features_text", ""), 
                             height=100, 
                             help="List of features for the ontology. Use CamelCase terms, optionally with context in parentheses (units, values, etc.).")
features = [f.strip() for f in features_text.split(",") if f.strip()]
st.session_state["features_text"] = features_text

if features and not question.strip():
    st.info("No research question provided. Generating a general ontology from the feature list.")

# Button: Generate Ontology
if st.button("Generate Ontology"):
    if not api_key:
        st.error("Please enter your OpenAI API key to generate the ontology.")
        st.stop()
    if not features:
        st.error("Feature list is empty. Please enter at least one feature or select a scenario.")
        st.stop()
    os.environ["OPENAI_API_KEY"] = api_key.strip()
    # Step 1: Parse feature context for values/units
    feature_context_map: Dict[str, str] = {}
    base_features: List[str] = []
    for feat in features:
        if "(" in feat and ")" in feat:
            term = feat.split("(", 1)[0].strip()
            context = feat.split("(", 1)[1].rstrip(")").strip()
            if term:
                base_features.append(term)
                if context:
                    feature_context_map[term] = context
        else:
            base_features.append(feat)
    features = base_features
    # Step 2: Find synonyms and merge
    syn_map = {}
    if features:
        if not quick_mode:
            syn_map = cached_find_synonyms(tuple(sorted(features)))
    if syn_map:
        merged_into: Dict[str, str] = {}
        for canon, alts in list(syn_map.items()):
            if canon not in syn_map:
                continue
            for alt in list(alts):
                if alt in syn_map and alt != canon:
                    alt_syns = syn_map.pop(alt, [])
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
    # Step 3: Build initial ontology structure
    with st.spinner("Classifying features and proposing relationships..."):
        structure = build_ontology_structure(question, features)
    if not structure:
        st.stop()
    # Step 4: Incorporate values and units from feature context
    if feature_context_map:
        for term, context in feature_context_map.items():
            if context:
                context = context.replace("ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡", "")
            quant_results = []
            try:
                from quantulum3 import parser
                quant_results = parser.parse(context)
            except Exception:
                quant_results = []
            if quant_results:
                for quant in quant_results:
                    if quant.value is not None:
                        val_str = str(quant.value).rstrip(".0")
                        if val_str:
                            structure.setdefault("values", [])
                            if val_str not in structure["values"]:
                                structure["values"].append(val_str)
                            rel_val = {"subject": term, "predicate": "hasValue", "object": val_str}
                            if rel_val not in structure.get("relationships", []):
                                structure.setdefault("relationships", []).append(rel_val)
                    if quant.unit:
                        unit_name = quant.unit.name or str(quant.unit.symbol)
                        if unit_name:
                            unit_term = to_valid_uri(unit_name)
                            if unit_term in ["C", "Degc", "Degreec"]:
                                unit_term = "DegreeCelsius"; st.session_state["unit_replacements"][unit_term] = unit_name
                            elif unit_term in ["F", "Degf", "Degreef"]:
                                unit_term = "DegreeFahrenheit"; st.session_state["unit_replacements"][unit_term] = unit_name
                            elif unit_term.lower() == "k" or unit_term.lower() == "kelvin":
                                unit_term = "Kelvin"; st.session_state["unit_replacements"][unit_term] = unit_name
                            elif unit_term.lower() == "atm" or unit_term == "Atmosphere":
                                unit_term = "Atmosphere"; st.session_state["unit_replacements"][unit_term] = unit_name
                            structure.setdefault("units", [])
                            if unit_term not in structure["units"]:
                                structure["units"].append(unit_term)
                            rel_unit = {"subject": term, "predicate": "hasUnit", "object": unit_term}
                            if rel_unit not in structure.get("relationships", []):
                                structure.setdefault("relationships", []).append(rel_unit)
                continue
            # Fallback to regex parsing if quantulum fails
            values_found = re.findall(r"\d+\.?\d*", context)
            values_found = [v for v in values_found if v != ""]
            context_clean = context
            for word in ["range", "Range", "typical", "typically", "Typical", "Typically", "approx", "approximately", "Approx", "Approximately"]:
                context_clean = context_clean.replace(word, "")
            unit_tokens = re.findall(r"[^\d\s,;()\-\u2013\u2014]+", context_clean)
            unit_tokens = [u for u in unit_tokens if u and not re.match(r"^[\W_]+$", u)]
            if 'e' in unit_tokens and re.search(r"\d+e[-+]?\d+", context, flags=re.IGNORECASE):
                unit_tokens = [u for u in unit_tokens if u.lower() != 'e']
            unit_tokens = list(dict.fromkeys(unit_tokens))
            for val in values_found:
                try:
                    float(val)
                except ValueError:
                    continue
                structure.setdefault("values", [])
                if val not in structure["values"]:
                    structure["values"] = structure.get("values", []) + [val]
                rel_val = {"subject": term, "predicate": "hasValue", "object": val}
                if rel_val not in structure.get("relationships", []):
                    structure.setdefault("relationships", []).append(rel_val)
            for u in unit_tokens:
                unit_name = u
                if u in ["C", "degC", "Ãƒâ€šÃ‚Â°C", "c"]:
                    unit_name = "DegreeCelsius"; st.session_state["unit_replacements"][unit_name] = u
                elif u in ["F", "degF", "Ãƒâ€šÃ‚Â°F", "f"]:
                    unit_name = "DegreeFahrenheit"; st.session_state["unit_replacements"][unit_name] = u
                elif u in ["K", "k"]:
                    unit_name = "Kelvin"; st.session_state["unit_replacements"][unit_name] = u
                elif u.lower() == "atm":
                    unit_name = "Atmosphere"; st.session_state["unit_replacements"][unit_name] = u
                else:
                    if len(u) == 1:
                        continue
                    unit_name = u
                structure.setdefault("units", [])
                if unit_name not in structure["units"]:
                    structure["units"].append(unit_name)
                rel_unit = {"subject": term, "predicate": "hasUnit", "object": unit_name}
                if rel_unit not in structure.get("relationships", []):
                    structure.setdefault("relationships", []).append(rel_unit)
    # (Removed Step 4.5/4.6: ensure domain coverage and reaction patterns are no longer needed due to scenario completeness)
    # Step 4.5: Ensure each Measurement has an Instrument and an Agent
    if structure.get("measurements"):
        if not structure.get("instruments"):
            structure["instruments"] = ["GenericInstrument"]
        if not structure.get("agents"):
            structure["agents"] = ["Experimenter"]
        for meas in structure["measurements"]:
            inst_rel_exists = any(r for r in structure.get("relationships", []) if r.get("predicate") == "usesInstrument" and r.get("subject") == meas)
            if not inst_rel_exists:
                structure["relationships"].append({"subject": meas, "predicate": "usesInstrument", "object": structure["instruments"][0]})
            agent_rel_exists = any(r for r in structure.get("relationships", []) if r.get("predicate") == "wasAssociatedWith" and r.get("subject") == meas)
            if not agent_rel_exists:
                structure["relationships"].append({"subject": meas, "predicate": "wasAssociatedWith", "object": structure["agents"][0]})
    # Step 5: Critique structure for questionable relations or missing elements
    feedback = critique_structure(structure)
    if feedback and feedback.lower() != "ok":
        md_box("### Critic Feedback", "warning")
        st.write(feedback)
        auto_fix = st.checkbox("Auto-fix critique (add/remove)", value=False)
        remove_flagged = st.checkbox("Remove flagged relations (per critic)", value=(not auto_fix))
        if auto_fix:
            fix_agent = Agent(
                role="Ontology Fixer",
                goal="Revise the ontology JSON by addressing the critique feedback",
                backstory="You are an expert ontologist who will remove incorrect triples and add missing ones as suggested by the critique.",
                llm=LLM(model="openai/gpt-4", temperature=0.0, max_tokens=1500)
            )
            fix_prompt = (
                "The following is an ontology represented as JSON and a critique of it. "
                "Remove any relationships flagged as incorrect in the critique, and add any missing relationships the critique suggests. "
                "Return only the full updated ontology JSON with the fixes applied.\n\n"
                "Ontology JSON:\n```json\n" + json.dumps(structure, indent=2) + "\n```\n"
                "Critique:\n" + feedback + "\n"
                "Output the revised ontology as JSON:\n"
            )
            fix_task = Task(description=fix_prompt, expected_output="Corrected ontology JSON", agent=fix_agent)
            Crew(agents=[fix_agent], tasks=[fix_task], process=Process.sequential).kickoff()
            raw_fix = str(fix_task.output).strip().strip("`")
            structure_new = None
            try:
                structure_new = json.loads(raw_fix)
            except json.JSONDecodeError:
                try:
                    structure_new = json.loads(raw_fix.replace("'", "\""))
                except Exception:
                    try:
                        structure_new = ast.literal_eval(raw_fix)
                    except Exception:
                        structure_new = None
            if isinstance(structure_new, dict):
                structure = structure_new
                md_box("Applied critic suggestions (auto-fix).", "info")
            else:
                st.error("Automatic critique fix failed. Please review and edit manually.")
        elif remove_flagged:
            flagged = []
            for rel in structure.get("relationships", []):
                rel_str = f"{rel.get('subject')} {rel.get('predicate')} {rel.get('object')}"
                for line in feedback.splitlines():
                    if rel_str in line:
                        flagged.append(rel)
                        break
            if flagged:
                structure["relationships"] = [r for r in structure["relationships"] if r not in flagged]
                md_box(f"Removed {len(flagged)} relation(s) per critic feedback.", "info")
    else:
        feedback = "OK"
    st.session_state["structure"] = structure
    st.session_state["synonyms_map"] = syn_map
    st.session_state["unit_replacements"] = st.session_state.get("unit_replacements", {})
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
        structure = st.session_state["structure"]
        structure["relationships"] = final_rels
        # Identify new terms introduced in edited relations
        existing_terms = sum([structure.get(cat, []) for cat in structure if isinstance(structure.get(cat), list)], [])
        introduced = {rel["subject"] for rel in final_rels if rel["subject"] not in existing_terms}
        introduced |= {rel["object"] for rel in final_rels if rel["object"] not in existing_terms}
        new_idents=[]; new_names=[]; new_measures=[]; new_instruments=[]; new_agents=[]; new_data=[]
        new_matters=[]; new_params=[]; new_values=[]; new_units=[]; new_properties=[]
        new_chemStressors=[]; new_radicals=[]
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
                    is_prop = any(r.get("object") == s and r.get("predicate") in {"hasProperty", "influences", "measures"} for r in final_rels)
                    if is_prop:
                        new_properties.append(s)
                    else:
                        new_params.append(s)
                    introduced.discard(s)
                if o in introduced:
                    new_values.append(o); introduced.discard(o)
            if p == "hasUnit":
                if s in introduced:
                    is_prop = any(r.get("object") == s and r.get("predicate") in {"hasProperty", "influences", "measures"} for r in final_rels)
                    if is_prop:
                        new_properties.append(s)
                    else:
                        new_params.append(s)
                    introduced.discard(s)
                if o in introduced:
                    new_units.append(o); introduced.discard(o)
            if p == "influences":
                if s in introduced:
                    low = s.lower()
                    if "radical" in low:
                        new_radicals.append(s)
                    elif any(x in low for x in ["peroxide", "acid"]):
                        new_chemStressors.append(s)
                    elif any(x in s for x in ["Test", "Experiment", "Protocol"]):
                        new_measures.append(s)
                    else:
                        new_params.append(s)
                    introduced.discard(s)
                if o in introduced:
                    if o not in new_properties:
                        new_properties.append(o)
                    introduced.discard(o)
            if p == "hasProperty":
                if s in introduced:
                    new_matters.append(s); introduced.discard(s)
                if o in introduced:
                    new_properties.append(o); introduced.discard(o)
        # Add new terms to appropriate categories
        if new_idents:
            structure.setdefault("identifiers", []).extend(x for x in new_idents if x not in structure.get("identifiers", []))
        if new_names:
            structure.setdefault("names", []).extend(x for x in new_names if x not in structure.get("names", []))
        if new_measures:
            structure.setdefault("measurements", []).extend(x for x in new_measures if x not in structure.get("measurements", []))
        if new_instruments:
            structure.setdefault("instruments", []).extend(x for x in new_instruments if x not in structure.get("instruments", []))
        if new_agents:
            structure.setdefault("agents", []).extend(x for x in new_agents if x not in structure.get("agents", []))
        if new_data:
            structure.setdefault("data", []).extend(x for x in new_data if x not in structure.get("data", []))
        if new_matters:
            structure.setdefault("matters", []).extend(x for x in new_matters if x not in structure.get("matters", []))
        if new_params:
            structure.setdefault("parameters", []).extend(x for x in new_params if x not in structure.get("parameters", []))
        if new_values:
            structure.setdefault("values", []).extend(x for x in new_values if x not in structure.get("values", []))
        if new_units:
            structure.setdefault("units", []).extend(x for x in new_units if x not in structure.get("units", []))
        if new_properties:
            structure.setdefault("properties", []).extend(x for x in new_properties if x not in structure.get("properties", []))
        if new_chemStressors:
            structure.setdefault("chemicalStressors", []).extend(x for x in new_chemStressors if x not in structure.get("chemicalStressors", []))
        if new_radicals:
            structure.setdefault("radicalSpecies", []).extend(x for x in new_radicals if x not in structure.get("radicalSpecies", []))
        if introduced:
            structure.setdefault("metadata", []).extend(x for x in introduced if x not in structure.get("metadata", []))
        # Deduplicate all category lists
        for cat in ["matters","properties","parameters","manufacturings","measurements","instruments","agents","data","metadata","identifiers","names","values","units","chemicalStressors","radicalSpecies","degradationObservables","testProtocols","reactionPathways"]:
            if cat in structure and isinstance(structure[cat], list):
                structure[cat] = list(dict.fromkeys(structure[cat]))
        st.session_state["structure"] = structure
        st.session_state["relation_review_done"] = True
        st.success("Relationships confirmed. Finalizing ontology...")

# Finalize ontology generation and output
if st.session_state.get("ontology_generated") and st.session_state.get("relation_review_done"):
    structure = st.session_state["structure"]
    syn_map = st.session_state.get("synonyms_map", {})
    unit_replacements = st.session_state.get("unit_replacements", {})
    # Step 6: Adjust categories for consistency based on relationships
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
            if s not in structure.get("parameters", []) and s not in structure.get("manufacturings", []) and s not in structure.get("measurements", []) and s not in structure.get("chemicalStressors", []) and s not in structure.get("radicalSpecies", []):
                cat_guess = guess_category(s)
                if cat_guess:
                    add_to_category(cat_guess, s)
                else:
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
            if s not in structure.get("matters", []) and s not in structure.get("manufacturings", []) and s not in structure.get("measurements", []) and s not in structure.get("reactionPathways", []):
                add_to_category("matters", s)
            if o not in structure.get("matters", []) and o not in structure.get("manufacturings", []) and o not in structure.get("measurements", []) and o not in structure.get("reactionPathways", []):
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
                # If manufacturing category exists, assume process likely manufacturing
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
                is_prop = any(r.get("object") == s and r.get("predicate") in {"hasProperty", "influences", "measures"} for r in structure.get("relationships", []))
                add_to_category("properties" if is_prop else "parameters", s)
            if o not in structure.get("values", []):
                add_to_category("values", o)
        elif p == "hasUnit":
            if s not in structure.get("parameters", []) and s not in structure.get("properties", []):
                is_prop = any(r.get("object") == s and r.get("predicate") in {"hasProperty", "influences", "measures"} for r in structure.get("relationships", []))
                add_to_category("properties" if is_prop else "parameters", s)
            if o not in structure.get("units", []):
                add_to_category("units", o)
    if adjustments:
        st.info("**Category Adjustments:** " + "; ".join(adjustments))
    # Deduplicate all category lists
    for cat in ["matters","properties","parameters","manufacturings","measurements","instruments","agents","data","metadata","identifiers","names","values","units","chemicalStressors","radicalSpecies","degradationObservables","testProtocols","reactionPathways"]:
        if cat in structure and isinstance(structure[cat], list):
            structure[cat] = list(dict.fromkeys(structure[cat]))
    # Step 7: Load base ontology if available
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
    # Step 8: Assemble ontology graph (RDF)
    with st.spinner("Assembling ontology graph..."):
        G, base_ns = assemble_ontology_graph(structure, base_graph)
    # Step 9: Align classes with external ontologies
    class_cat_pairs = []
    for cat_key, display_name in [
        ("matters","Matter"), 
        ("properties","Property"), 
        ("parameters","Parameter"), 
        ("manufacturings","Manufacturing process"), 
        ("measurements","Measurement process"), 
        ("instruments","Instrument"), 
        ("agents","Agent"), 
        ("data","Data"), 
        ("identifiers","Identifier"), 
        ("names","Name"), 
        ("values","Value"), 
        ("units","Unit"), 
        ("chemicalStressors","Chemical substance"), 
        ("radicalSpecies","Radical species"), 
        ("degradationObservables","Observable property"), 
        ("testProtocols","Test protocol"), 
        ("reactionPathways","Reaction pathway")
    ]:
        for name in structure.get(cat_key, []):
            class_cat_pairs.append((name, display_name))
    align_map = {}
    if class_cat_pairs:
        with st.spinner("Aligning classes with external ontologies..."):
            align_map = map_to_external(class_cat_pairs)
            # Add known unit mappings if not found via GPT/BioPortal
            for unit_name, symbol in unit_replacements.items():
                if unit_name and unit_name not in align_map:
                    key = to_valid_uri(unit_name)
                    if key in {
                        "DegreeCelsius","DegreeFahrenheit","Kelvin","Atmosphere","Bar",
                        "Pascal","Percent","Second","Minute","Hour"
                    }:
                        known_uri = {
                            "DegreeCelsius": "http://qudt.org/vocab/unit#DegreeCelsius",
                            "DegreeFahrenheit": "http://qudt.org/vocab/unit#DegreeFahrenheit",
                            "Kelvin": "http://qudt.org/vocab/unit#Kelvin",
                            "Atmosphere": "http://qudt.org/vocab/unit#Atmosphere",
                            "Bar": "http://qudt.org/vocab/unit#Bar",
                            "Pascal": "http://qudt.org/vocab/unit#Pascal",
                            "Percent": "http://qudt.org/vocab/unit#Percent",
                            "Second": "http://qudt.org/vocab/unit#Second",
                            "Minute": "http://qudt.org/vocab/unit#Minute",
                            "Hour": "http://qudt.org/vocab/unit#Hour"
                        }.get(key)
                        if known_uri:
                            align_map[unit_name] = {"ontology": "QUDT Unit", "uri": known_uri}
            add_external_mappings(G, align_map, base_ns)
    if align_map:
        align_info = [f"{cls} ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“ *{info.get('ontology')}* (`{info.get('uri')}`)" for cls, info in align_map.items()]
        st.write("**External Alignments:** " + "; ".join(align_info))
    else:
        st.write("**External Alignments:** No direct matches found.")
    # Step 10: Add labels (prefLabel, altLabel) and definitions (comments)
    for canon, symbol in unit_replacements.items():
        syn_map.setdefault(canon, [])
        if symbol not in syn_map[canon]:
            syn_map[canon].append(symbol)
    with st.spinner("Adding labels and definitions..."):
        if not quick_mode:
            add_labels_and_definitions(G, syn_map, structure, base_ns)
        else:
            SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
            for cat, items in structure.items():
                if cat == "relationships" or not isinstance(items, list):
                    continue
                for name in items:
                    if not isinstance(name, str):
                        continue
                    uri = base_ns[to_valid_uri(name)]
                    label_text = re.sub(CAMEL_RE, " ", name).strip()
                    G.add((uri, SKOS.prefLabel, Literal(label_text, lang="en")))
                    for alt in syn_map.get(name, []):
                        alt_label = re.sub(CAMEL_RE, " ", alt).strip()
                        if alt_label and alt_label.lower() != label_text.lower():
                            G.add((uri, SKOS.altLabel, Literal(alt_label, lang="en")))
    # Step 11: Apply versioning metadata to ontology
    apply_versioning(G, base_graph)
    # Step 12: Generate provenance pattern for measurements (PROV Activity equivalent classes)
    for rel in structure.get("relationships", []):
        if rel.get("predicate") == "measures":
            meas_name = rel.get("subject"); prop_name = rel.get("object")
            inst_name = None; agent_name = None
            for r2 in structure.get("relationships", []):
                if r2.get("subject") == meas_name and r2.get("predicate") == "usesInstrument":
                    inst_name = r2.get("object")
                if r2.get("subject") == meas_name and r2.get("predicate") == "wasAssociatedWith":
                    agent_name = r2.get("object")
            if not inst_name and structure.get("instruments"):
                inst_name = structure["instruments"][0]
            if not agent_name and structure.get("agents"):
                agent_name = structure["agents"][0]
            if inst_name and agent_name:
                meas_cls = base_ns[to_valid_uri(meas_name)]
                prop_cls = base_ns[to_valid_uri(prop_name)]
                inst_cls = base_ns[to_valid_uri(inst_name)]
                agent_cls = base_ns[to_valid_uri(agent_name)]
                activity = BNode()
                G.add((activity, RDF.type, URIRef("http://www.w3.org/ns/prov#Activity")))
                G.add((activity, base_ns.measures, prop_cls))
                G.add((activity, base_ns.usesInstrument, inst_cls))
                G.add((activity, URIRef("http://www.w3.org/ns/prov#wasAssociatedWith"), agent_cls))
                G.add((meas_cls, OWL.equivalentClass, activity))
    # Step 13: Generate SHACL shapes and validate (optional auto-fix)
    shapes_agent = Agent(
        role="SHACL Builder",
        goal="Create SHACL NodeShapes to enforce ontology constraints",
        backstory="You are an ontology engineer writing SHACL shapes to validate the ontology's consistency.",
        llm=LLM(model="openai/gpt-4", temperature=0.0, max_tokens=800)
    )
    shapes_prompt = (
        "Generate SHACL shape definitions (in Turtle) that enforce the following constraints:\n"
        "ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢ Each qudt:QuantityValue must have exactly 1 qudt:numericValue (xsd:double) and exactly 1 qudt:unit.\n"
        "ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢ The property mat:hasProperty should have domain Matter (or subclasses) and range Property (or subclasses).\n"
        "ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢ Each Measurement process must have at least 1 mat:measures Property.\n"
        "ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢ Each Measurement process must have at least 1 mat:usesInstrument Instrument.\n"
        "ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢ Each Measurement process must have at least 1 prov:wasAssociatedWith Agent.\n"
        "Return only Turtle syntax for the SHACL NodeShapes."
    )
    conforms = True
    report_graph = None
    shape_ttl = ""
    if not quick_mode:
        shape_task = Task(description=shapes_prompt, expected_output="Turtle SHACL shapes", agent=shapes_agent)
        Crew(agents=[shapes_agent], tasks=[shape_task], process=Process.sequential).kickoff()
        shape_ttl = str(shape_task.output).strip().strip("`")
        try:
            G.parse(data=shape_ttl, format="turtle")
        except Exception:
            st.warning("SHACL shapes could not be parsed. Proceeding without shapes.")
        try:
            conforms, report_graph, report_text = validate(data_graph=G, shacl_graph=None, inference='rdfs')
        except Exception as e:
            st.error(f"SHACL validation error: {e}")
            conforms = True
    if not conforms:
        st.error("SHACL validation failed; see report below.")
        report_ttl = report_graph.serialize(format='turtle') if report_graph else b''
        st.text(report_ttl.decode('utf-8') if isinstance(report_ttl, bytes) else report_ttl)
        st.download_button("Download SHACL report", report_ttl, file_name='shacl_report.ttl', mime='text/turtle')
        if st.button("Fix violations automatically"):
            fix_attempt = 0
            fixed = False
            current_ttl = G.serialize(format="turtle")
            start_time = time.time()
            while fix_attempt < 3 and not fixed:
                fix_attempt += 1
                fix_agent = Agent(
                    role="Ontology Fixer",
                    goal="Apply minimal changes to fix SHACL violations",
                    backstory="You are an expert ontologist who will correct the ontology to satisfy the SHACL constraints.",
                    llm=LLM(model="openai/gpt-4", temperature=0.0, max_tokens=1500)
                )
                fix_prompt = (
                    "The following ontology (including SHACL shapes) has validation errors. "
                    "Based on the SHACL report, modify the ontology with minimal changes (adding or removing triples) to fix all violations. "
                    "Return only the full corrected ontology in Turtle format.\n\n"
                    "Ontology Turtle:\n```turtle\n" + (current_ttl.decode('utf-8') if isinstance(current_ttl, bytes) else current_ttl) + "\n```\n"
                    "SHACL Report:\n```turtle\n" + (report_graph.serialize(format='turtle').decode('utf-8') if report_graph else '') + "\n```"
                )
                fix_task = Task(description=fix_prompt, expected_output="Corrected ontology Turtle", agent=fix_agent)
                Crew(agents=[fix_agent], tasks=[fix_task], process=Process.sequential).kickoff()
                fixed_ttl = str(fix_task.output).strip().strip("`")
                try:
                    newG = Graph()
                    newG.parse(data=fixed_ttl, format="turtle")
                except Exception:
                    continue
                try:
                    c, new_report, _ = validate(data_graph=newG, shacl_graph=None, inference='rdfs')
                except Exception:
                    c = False
                if c:
                    G = newG
                    fixed = True
                    st.success("SHACL violations fixed by automatic patch.")
                    break
                else:
                    current_ttl = fixed_ttl.encode('utf-8') if isinstance(fixed_ttl, str) else fixed_ttl
                    report_graph = new_report
                if time.time() - start_time > 60:
                    st.error("Aborting SHACL auto-fix due to time/complexity limit.")
                    break
            if not fixed:
                st.error("Automatic fixing failed to produce a valid ontology. Please review manually.")
    # Final output: Provide download links and visualization
    ttl_data = None
    jsonld_data = None
    try:
        ttl_data = G.serialize(format="turtle")
    except Exception as e:
        st.error(f"Error serializing ontology to Turtle: {e}")
    if ttl_data:
        st.download_button("Download Ontology (Turtle)", ttl_data, file_name="hydrogen_ontology.ttl", mime="text/turtle")
    try:
        jsonld_data = G.serialize(format="json-ld", indent=2)
    except Exception as e:
        st.error(f"Error serializing ontology to JSON-LD: {e}")
    if jsonld_data:
        st.download_button("Download Ontology (JSON-LD)", jsonld_data, file_name="hydrogen_ontology.jsonld", mime="application/ld+json")
    try:
        nxG = nx.MultiDiGraph()
        base_nodes = ["Matter","Property","Parameter","Process"]
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
        if structure.get("chemicalStressors"): base_nodes.append("ChemicalStressor")
        if structure.get("radicalSpecies"): base_nodes.append("RadicalSpecies")
        if structure.get("degradationObservables"): base_nodes.append("DegradationObservable")
        if structure.get("testProtocols"): base_nodes.append("TestProtocol")
        if structure.get("reactionPathways"): base_nodes.append("ReactionPathway")
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
            ("metadata", "Metadata"),
            ("chemicalStressors", "ChemicalStressor"),
            ("radicalSpecies", "RadicalSpecies" if structure.get("radicalSpecies") else "ChemicalStressor"),
            ("degradationObservables", "DegradationObservable"),
            ("testProtocols", "TestProtocol" if structure.get("testProtocols") else ("Measurement" if structure.get("measurements") else "Process")),
            ("reactionPathways", "ReactionPathway")
        ]:
            for item in structure.get(key, []):
                node = to_valid_uri(item)
                cat = parent
                nxG.add_node(node, category=cat, is_base=False)
                if parent and parent != "Metadata":
                    nxG.add_edge(node, parent, label="isA")
        # Connect base category hierarchy
        if "ChemicalStressor" in nxG and "Matter" in nxG: nxG.add_edge("ChemicalStressor", "Matter", label="isA")
        if "RadicalSpecies" in nxG and "ChemicalStressor" in nxG: nxG.add_edge("RadicalSpecies", "ChemicalStressor", label="isA")
        if "DegradationObservable" in nxG and "Property" in nxG: nxG.add_edge("DegradationObservable", "Property", label="isA")
        if "TestProtocol" in nxG:
            if "Measurement" in nxG: nxG.add_edge("TestProtocol", "Measurement", label="isA")
            else: nxG.add_edge("TestProtocol", "Process", label="isA")
        if "ReactionPathway" in nxG and "Process" in nxG: nxG.add_edge("ReactionPathway", "Process", label="isA")
        # Generate GraphML and provide download
        graphml_data = nx.generate_graphml(nxG)
        graphml_str = "\n".join(list(graphml_data))
        st.download_button("Download Graph (GraphML)", graphml_str, file_name="hydrogen_ontology.graphml", mime="application/xml")
    except Exception as e:
        st.error(f"Error generating GraphML: {e}")
    # Graph visualization with Graphviz
    try:
        st.subheader("Ontology Graph Visualization")
        dot = Digraph(engine="dot", graph_attr={"rankdir": "TB"})
        shape_map = {
            "Matter": "box", "Property": "ellipse", "Parameter": "diamond", 
            "Process": "polygon", "Manufacturing": "polygon", "Measurement": "polygon",
            "Instrument": "parallelogram", "Agent": "hexagon", "Data": "cylinder",
            "Metadata": "note", "Identifier": "note", "Name": "note",
            "Value": "note", "Unit": "note",
            "ChemicalStressor": "box", "RadicalSpecies": "box", "DegradationObservable": "ellipse", 
            "TestProtocol": "polygon", "ReactionPathway": "polygon",
            "": "oval"
        }
        color_map = {
            "Matter": "#ffdede", "Property": "#deffde", "Parameter": "#dedeff",
            "Process": "#fff2cc", "Manufacturing": "#fff2cc", "Measurement": "#fff2cc",
            "Instrument": "#ffdcb3", "Agent": "#e0ccff", "Data": "#ccffeb",
            "Metadata": "#f0f0f0", "Identifier": "#e0ffff", "Name": "#e0e0ff",
            "Value": "#fff0e0", "Unit": "#ffe0f0",
            "ChemicalStressor": "#ffcccc", "RadicalSpecies": "#ffb3b3", "DegradationObservable": "#ccffcc",
            "TestProtocol": "#fff2cc", "ReactionPathway": "#d6f5ff",
            "": "#ffffff"
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
