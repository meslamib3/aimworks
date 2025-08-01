import streamlit as st
import os
import re
import json
import io

# ------------------------------- UI --------------------------------
st.title("Ontology Generation Pipeline")
st.write(
    "This Streamlit app generates an ontology extension for a given research question and list of features. "
    "It leverages GPT-4 (via the OpenAI API) and CrewAI agents to propose ontology classes and relationships, "
    "then constructs an extended ontology in Turtle and GraphML formats and visualises the graph."
)

question        = st.text_input("Research Question:")
features_input  = st.text_area("List of Features (comma-separated):")
api_key_input   = st.text_input("OpenAI API Key (required for GPT-4):", type="password")
features        = [f.strip() for f in features_input.split(",") if f.strip()]

# -------------------------- main button ----------------------------
if st.button("Generate Ontology"):
    if not question:  st.error("Please enter a research question."); st.stop()
    if not features:  st.error("Please enter at least one feature."); st.stop()
    if api_key_input: os.environ["OPENAI_API_KEY"] = api_key_input

    with st.spinner("Contacting GPT-4 and generating ontology …"):
        # dynamic imports
        try:
            import openai
            from crewai import Agent, Task, Crew, Process
            from crewai.llm import LLM
            from rdflib import Graph, Namespace, RDF, RDFS, OWL
            import networkx as nx
            import requests
        except ImportError as e:
            st.error(f"Missing library ? {e}.  Install dependencies and retry."); st.stop()

        # ------------------ helper utilities -------------------
        def to_valid_uri(name: str) -> str:
            parts = re.split(r"\W+", name)
            return "".join(p.capitalize() for p in parts if p) or name

        def fetch_base_ontology() -> Graph | None:
            urls = [
                "https://raw.githubusercontent.com/MaxDreger92/MatGraph/master/Ontology/MatGraphOntology.ttl",
                "https://raw.githubusercontent.com/MaxDreger92/MatGraph/master/Ontology/MatGraphOntology.owl",
            ]
            g = Graph()
            for url in urls:
                try:
                    data = requests.get(url, timeout=10).content
                    g.parse(data=data, format="turtle" if url.endswith(".ttl") else "xml")
                    return g
                except Exception:
                    continue
            return None

        # --------------- ontology construction ----------------
        def build_extended_ontology(data: dict, base_graph: Graph | None):
            g = Graph()
            if base_graph: g += base_graph

            # namespace
            base_ns = None
            if base_graph:
                for s in base_graph.subjects(RDF.type, OWL.Ontology):
                    base_ns = Namespace(str(s) + "#"); break
            if not base_ns: base_ns = Namespace("http://example.org/matgraph#")
            g.bind("mat", base_ns); g.bind("owl", OWL); g.bind("rdfs", RDFS); g.bind("rdf", RDF)

            # ensure core classes
            def ensure(c): uri = base_ns[c]; g.add((uri, RDF.type, OWL.Class)); return uri
            Material = ensure("Material"); Property = ensure("Property")
            Parameter = ensure("Parameter"); ProcessC = ensure("Process")
            Measurement   = ensure("Measurement"); g.add((Measurement, RDFS.subClassOf, ProcessC))
            Manufacturing = ensure("Manufacturing"); g.add((Manufacturing, RDFS.subClassOf, ProcessC))
            Metadata      = ensure("Metadata")

            # new subclasses
            for key, parent in [
                ("materials", Material),
                ("properties", Property),
                ("parameters", Parameter),
                ("processes", ProcessC),
                ("measurements", Measurement),
                ("manufacturing", Manufacturing),
                ("metadata", Metadata),
            ]:
                for n in data.get(key, []):
                    c = base_ns[to_valid_uri(n)]
                    g.add((c, RDF.type, OWL.Class)); g.add((c, RDFS.subClassOf, parent))

            # object properties
            hasProp = base_ns.hasProperty; influences = base_ns.influences
            g.add((hasProp, RDF.type, OWL.ObjectProperty)); g.add((hasProp, RDFS.range, Property))
            g.add((influences, RDF.type, OWL.ObjectProperty)); g.add((influences, RDFS.range, Property))

            # relationships
            for rel in data.get("relationships", []):
                s = base_ns[to_valid_uri(rel["subject"])]
                o = base_ns[to_valid_uri(rel["object"])]
                p = hasProp if rel["predicate"] == "hasProperty" else influences
                g.add((s, p, o))

            # networkx
            G = nx.MultiDiGraph()
            base_nodes = {
                Material:"Material", Property:"Property", Parameter:"Parameter",
                ProcessC:"Process", Measurement:"Process", Manufacturing:"Process",
                Metadata:"Metadata",
            }
            for uri, cat in base_nodes.items():
                G.add_node(uri.split("#")[-1], cat=cat, base=1)
            for cls, _, parent in g.triples((None, RDFS.subClassOf, None)):
                G.add_node(cls.split("#")[-1], cat=parent.split("#")[-1], base=0)
                G.add_edge(cls.split("#")[-1], parent.split("#")[-1], label="subClassOf")
            for s,p,o in g.triples((None, hasProp, None)):
                G.add_edge(s.split("#")[-1], o.split("#")[-1], label="hasProperty")
            for s,p,o in g.triples((None, influences, None)):
                G.add_edge(s.split("#")[-1], o.split("#")[-1], label="influences")
            return g, G

        # ---------------- LLM / CrewAI part ----------------
        def generate_structure(q: str, feats: list[str]) -> dict:
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is missing")

            llm_d = LLM(model="openai/gpt-4", temperature=0.3, max_tokens=1000)
            llm_o = LLM(model="openai/gpt-4", temperature=0.0, max_tokens=1000)

            domain_agent = Agent(
                role="Domain Expert",
                goal="Identify ontology classes",
                backstory="A materials scientist fluent in EMMO.",
                llm=llm_d,
            )
            ont_agent = Agent(
                role="Ontology Engineer",
                goal="Return JSON ontology extension",
                backstory="An ontology engineer.",
                llm=llm_o,
            )

            feats_txt = ", ".join(feats)
            domain_prompt = (
                f'Question: "{q}"\nFeatures: {feats_txt}\n'
                "Classify each feature as Material, Property, Parameter, Process, Measurement, Manufacturing, or Metadata. "
                "Suggest relationships using 'hasProperty' (Material?Property) and 'influences' (Process/Measurement/Manufacturing/Parameter?Property). "
                "Return structured lists."
            )
            ont_prompt = (
                "Convert the analysis into pure JSON with keys "
                '"materials", "properties", "parameters", "processes", '
                '"measurements", "manufacturing", "metadata", "relationships".'
            )

            d_task = Task(
                description=domain_prompt,
                expected_output="Lists of classes and relationships.",
                agent=domain_agent,
            )
            o_task = Task(
                description=ont_prompt,
                expected_output="JSON object with the specified keys.",
                agent=ont_agent,
                context=[d_task],
            )
            Crew([domain_agent, ont_agent], [d_task, o_task], process=Process.sequential).kickoff()

            raw = str(o_task.output).strip("` \n")
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSON parse error: {e}\nRaw output:\n{raw}")

        # --------------- pipeline execution ---------------
        try:
            ont_json = generate_structure(question, features)
        except Exception as e:
            st.error(f"Ontology generation failed: {e}"); st.stop()

    # ----------- build and visualise graph ---------------
    with st.spinner("Building ontology graph …"):
        base = fetch_base_ontology()
        if not base: st.warning("Base ontology not found; using minimal core.")
        try:
            ttl_graph, nx_graph = build_extended_ontology(ont_json, base)
        except Exception as e:
            st.error(f"Graph build error: {e}"); st.stop()

    st.success("Ontology generation complete! ?")

    st.download_button("Download Turtle (.ttl)",
                       ttl_graph.serialize(format="turtle"),
                       "extended_ontology.ttl",
                       "text/turtle")

    buf = io.BytesIO(); nx.write_graphml(nx_graph, buf)
    st.download_button("Download GraphML",
                       buf.getvalue().decode(),
                       "extended_ontology.graphml",
                       "application/xml")

    try:
        from graphviz import Digraph
        dot = Digraph()
        dot.attr(rankdir="TB")
        for n,a in nx_graph.nodes(data=True):
            style = {"style":"filled","fillcolor":"lightgray"} if a.get("base") else {}
            dot.node(n, **style)
        for u,v,ed in nx_graph.edges(data=True):
            dot.edge(u, v, label=ed.get("label",""))
        st.subheader("Ontology Graph")
        st.graphviz_chart(dot.source)
    except Exception as e:
        st.error(f"Graphviz error: {e}")
