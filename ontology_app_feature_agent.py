import streamlit as st
import os
import re
import json
import io
from typing import List, Tuple, Optional

# -------------------------------------------------
# ?? Streamlit layout
# -------------------------------------------------

st.set_page_config(page_title="Ontology Generation Pipeline", page_icon="??", layout="centered")

st.title("Ontology Generation Pipeline")
st.write(
    "Generate an ontology extension for a hydrogen-technology research question. "
    "An AI scientist agent first proposes a feature set; you can edit it, then build the ontology, download TTL/GraphML, and view the graph."
)

# -------------------------------------------------
# ?? Session state helpers
# -------------------------------------------------

def init_state():
    if "features" not in st.session_state:
        st.session_state["features"] = ""

init_state()

# -------------------------------------------------
# ?? User inputs
# -------------------------------------------------
question: str = st.text_input("Research Question:")
api_key: str = st.text_input("OpenAI API Key (required for GPT-4):", type="password")

# -------------------------------------------------
# ?? Feature-suggestion agent
# -------------------------------------------------

def suggest_features(question_text: str) -> str:
    """Return a comma-separated feature list via CrewAI agent."""
    if not question_text:
        st.error("Please enter a research question first.")
        return ""

    try:
        import openai
        from crewai import Agent, Task, Crew, Process
        from crewai.llm import LLM
    except ImportError as e:
        st.error(f"Missing library: {e}. Install it with `pip install {e.name}`.")
        return ""

    if not os.getenv("OPENAI_API_KEY"):
        st.error("Set your OpenAI API key then retry.")
        return ""

    openai.api_key = os.getenv("OPENAI_API_KEY")

    llm = LLM(model="openai/gpt-4", temperature=0.2, max_tokens=512)

    scientist = Agent(
        role="Hydrogen-Feature Scientist",
        goal="List all relevant experimental, material, and environmental features for hydrogen-technology research questions",
        backstory="A veteran researcher with decades of laboratory experience designing hydrogen storage and membrane experiments.",
        llm=llm,
    )

    prompt = (
        "Provide ONLY a comma-separated list (no newlines, no bullets) of 5-20 concise feature identifiers relevant to the research question below. "
        "Use CamelCase where appropriate.\n\n"
        f"Research Question: \"{question_text}\""
    )

    task = Task(description=prompt, expected_output="Comma-separated feature list", agent=scientist)
    crew = Crew(agents=[scientist], tasks=[task], process=Process.sequential)
    crew.kickoff()

    output = str(task.output).strip()
    if output.startswith("```"):
        output = output.strip("`\n ")
    return output

# Suggest features button
if st.button("Suggest Features"):
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    st.session_state["features"] = suggest_features(question)

# Editable feature list
features_input: str = st.text_area(
    "List of Features (comma-separated):",
    value=st.session_state.get("features", ""),
    height=120,
)
# Keep state in sync
st.session_state["features"] = features_input

features: List[str] = [f.strip() for f in features_input.split(",") if f.strip()]

# -------------------------------------------------
# ?? Ontology helpers
# -------------------------------------------------

def to_valid_uri(name: str) -> str:
    parts = re.split(r"\W+", name)
    return "".join(p.capitalize() for p in parts if p) or name

# -------------------------------------------------
# ? Generate Ontology button
# -------------------------------------------------
if st.button("Generate Ontology"):
    # ---- Validation ----
    if not question:
        st.error("Please enter a research question.")
        st.stop()
    if not features:
        st.error("Please provide at least one feature.")
        st.stop()
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    with st.spinner("Generating ontology structure via GPT-4…"):
        # Dynamic imports
        try:
            import openai
            from crewai import Agent, Task, Crew, Process
            from crewai.llm import LLM
            import networkx as nx
            from rdflib import Graph, Namespace, RDF, RDFS, OWL
            import requests
            from graphviz import Digraph
        except ImportError as e:
            st.error(f"Missing dependency: {e}. Install it and retry.")
            st.stop()

        # ---------- Fetch base ontology ----------
        def fetch_base_ontology() -> Optional["Graph"]:
            urls = [
                "https://raw.githubusercontent.com/MaxDreger92/MatGraph/master/Ontology/MatGraphOntology.ttl",
                "https://raw.githubusercontent.com/MaxDreger92/MatGraph/master/Ontology/MatGraphOntology.owl",
            ]
            g = Graph()
            for url in urls:
                try:
                    r = requests.get(url, timeout=10)
                    if r.status_code == 200:
                        fmt = "turtle" if url.endswith("ttl") else "xml"
                        g.parse(data=r.content, format=fmt)
                        return g
                except Exception:
                    continue
            return None

        # ---------- Build extended ontology ----------
        def build_extended_ontology(data: dict, base: Optional["Graph"]) -> Tuple["Graph", "nx.MultiDiGraph"]:
            graph = Graph()
            if base:
                graph += base
            base_ns = None
            if base:
                for subj in base.subjects(RDF.type, OWL.Ontology):
                    base_ns = Namespace(str(subj) + "#")
                    break
            base_ns = base_ns or Namespace("http://example.org/matgraph#")
            graph.bind("mat", base_ns)
            graph.bind("owl", OWL)
            graph.bind("rdfs", RDFS)
            graph.bind("rdf", RDF)

            # Core classes
            def ensure_class(uri):
                if (uri, RDF.type, OWL.Class) not in graph:
                    graph.add((uri, RDF.type, OWL.Class))
            material = base_ns.Material; ensure_class(material)
            prop = base_ns.Property; ensure_class(prop)
            param = base_ns.Parameter; ensure_class(param)
            proc = base_ns.Process; ensure_class(proc)

            # Add subclasses
            for key, parent in [
                ("materials", material),
                ("properties", prop),
                ("parameters", param),
                ("processes", proc),
            ]:
                for item in data.get(key, []):
                    cls = base_ns[to_valid_uri(item)]
                    ensure_class(cls)
                    graph.add((cls, RDFS.subClassOf, parent))

            # Object Properties
            hasP = base_ns.hasProperty; influences = base_ns.influences
            if (hasP, RDF.type, OWLObject := OWL.ObjectProperty) not in graph:
                graph.add((hasP, RDF.type, OWLObject))
                graph.add((hasP, RDFS.domain, material)); graph.add((hasP, RDFS.range, prop))
            if (influences, RDF.type, OWLObject) not in graph:
                graph.add((influences, RDF.type, OWLObject)); graph.add((influences, RDFS.range, prop))

            for rel in data.get("relationships", []):
                subj = base_ns[to_valid_uri(rel["subject"])]
                pred_uri = hasP if rel["predicate"] == "hasProperty" else influences
                obj = base_ns[to_valid_uri(rel["object"])]
                graph.add((subj, pred_uri, obj))

            # Build NetworkX graph for viz
            G = nx.MultiDiGraph()
            for node, cat in {
                material: "Material", prop: "Property", param: "Parameter", proc: "Process",
            }.items():
                label = str(node).split("#")[-1]
                G.add_node(label, category=cat, is_base=1)
            for key, parent in [
                ("materials", material), ("properties", prop), ("parameters", param), ("processes", proc)
            ]:
                for item in data.get(key, []):
                    n = to_valid_uri(item); p = str(parent).split("#")[-1]
                    G.add_node(n, category=p, is_base=0); G.add_edge(n, p, label="subClassOf")
            for rel in data.get("relationships", []):
                s = to_valid_uri(rel["subject"]); o = to_valid_uri(rel["object"])
                G.add_node(s, category="", is_base=0); G.add_node(o, category="", is_base=0)
                G.add_edge(s, o, label=rel["predicate"])
            return graph, G

        # ---------- LLM multi-agent to classify & relate ----------
        def generate_structure(q: str, feats: List[str]) -> dict:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            llm_d = LLM(model="openai/gpt-4", temperature=0.3, max_tokens=1000)
            llm_e = LLM(model="openai/gpt-4", temperature=0.0, max_tokens=800)
            domain = Agent(role="Domain Expert", goal="Classify features", backstory="Ontologist", llm=llm_d)
            engineer = Agent(role="Ontology Engineer", goal="Emit JSON", backstory="Formats output", llm=llm_e)
            feat_text = ", ".join(feats)
            domain_prompt = (
                f"Question: {q}\nFeatures: {feat_text}\n\n"  # noqa
                "Categorize each feature as Material, Property, Parameter, or Process. "
                "Suggest relationships using hasProperty (Material?Property) or influences (Parameter/Process?Property)."
            )
            ont_prompt = (
                "Produce ONLY JSON with keys materials, properties, parameters, processes, relationships. "
                "relationships is a list of objects {subject,predicate,object}."
            )
            t1 = Task(description=domain_prompt, expected_output="Class list", agent=domain)
            t2 = Task(description=ont_prompt, expected_output="JSON", agent=engineer, context=[t1])
            Crew(agents=[domain, engineer], tasks=[t1, t2], process=Process.sequential).kickoff()
            raw = str(t2.output).strip()
            return json.loads(raw.strip("`\n "))

        # ---------- Run pipeline ----------
        try:
            structure = generate_structure(question, features)
        except Exception as e:
            st.error(f"Ontology structure generation failed: {e}")
            st.stop()

    # Build graphs
    with st.spinner("Building ontology graph…"):
        base_g = fetch_base_ontology()
        if base_g is None:
            st.warning("Base ontology not found; using minimal core classes.")
        try:
            extended_g, nx_graph = build_extended_ontology(structure, base_g)
        except Exception as e:
            st.error(f"Error constructing ontology: {e}"); st.stop()

    st.success("Ontology generation complete!")

    # ---- Downloads ----
    try:
        turtle_str = extended_g.serialize(format="turtle")
        st.download_button("Download TTL", turtle_str, "extended_ontology.ttl", "text/turtle")
    except Exception as e:
        st.error(f"Turtle serialization failed: {e}")

    try:
        buffer = io.BytesIO(); nx.write_graphml(nx_graph, buffer)
        st.download_button("Download GraphML", buffer.getvalue().decode(), "extended_ontology.graphml", "application/xml")
    except Exception as e:
        st.error(f"GraphML generation failed: {e}")

    # ---- Visualization ----
    try:
        dot = Digraph(format="png", graph_attr={"rankdir": "TB"})
        for n, attrs in nx_graph.nodes(data=True):
            style = {True: ("lightgray", "filled"), False: (None, None)}[attrs.get("is_base") == 1]
            dot.node(n, n, shape="ellipse", fillcolor=style[0] or "", style=style[1] or "")
        for u, v, a in nx_graph.edges(data=True):
            dot.edge(u, v, label=a.get("label", ""))
        st.subheader("Ontology Graph Visualization")
        st.graphviz_chart(dot.source)
    except Exception as e:
        st.error
