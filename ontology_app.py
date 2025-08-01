import streamlit as st
import os
import re
import json

# Title and description
st.title("Ontology Generation Pipeline")
st.write(
    "This Streamlit app generates an ontology extension for a given research question and list of features. "
    "It uses GPT-4 via the OpenAI API and the CrewAI framework to propose ontology classes and relationships, "
    "then constructs an extended ontology in Turtle and GraphML formats, as well as a visual graph."
)

# Input fields for question, features, and API key
question = st.text_input("Research Question:")
features_input = st.text_area("List of Features (comma-separated):")
api_key_input = st.text_input("OpenAI API Key (required for GPT-4):", type="password")

# Parse the comma-separated features into a list
features = [f.strip() for f in features_input.split(",") if f.strip()]

# When the user clicks the button, trigger the ontology generation
if st.button("Generate Ontology"):
    # Basic validation
    if not question:
        st.error("Please enter a research question.")
        st.stop()
    if not features:
        st.error("Please enter at least one feature.")
        st.stop()
    # If an API key is provided via UI, set it as environment variable for OpenAI
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input

    # Begin generation process
    with st.spinner("Generating ontology structure via GPT-4 (this may take some time)..."):
        # Import required libraries and handle if not installed
        try:
            import openai
        except ImportError:
            st.error("The OpenAI library is not installed. Install it with `pip install openai`.")
            st.stop()
        try:
            from crewai import Agent, Task, Crew, Process
            from crewai.llm import LLM
        except ImportError:
            st.error("The CrewAI library is not installed. Install it with `pip install crewai`.")
            st.stop()
        try:
            import requests
        except ImportError:
            requests = None  # requests is optional for fetching base ontology
        try:
            from rdflib import Graph, Namespace, RDF, RDFS, OWL
        except ImportError:
            st.error("The rdflib library is not installed. Install it with `pip install rdflib`.")
            st.stop()
        try:
            import networkx as nx
        except ImportError:
            st.error("The networkx library is not installed. Install it with `pip install networkx`.")
            st.stop()

        # Helper functions from the original script
        def to_valid_uri(name: str) -> str:
            """Convert an arbitrary name string into a CamelCase URI fragment (remove special chars)."""
            parts = re.split(r'\W+', name)
            camel = ''.join(part.capitalize() for part in parts if part)
            return camel or name

        def fetch_base_ontology() -> Graph:
            """Fetch the base ontology from the MatGraph GitHub repository (if available)."""
            if not requests:
                return None
            base_graph = Graph()
            urls = [
                "https://raw.githubusercontent.com/MaxDreger92/MatGraph/master/Ontology/MatGraphOntology.ttl",
                "https://raw.githubusercontent.com/MaxDreger92/MatGraph/master/Ontology/MatGraphOntology.owl"
            ]
            for url in urls:
                try:
                    resp = requests.get(url, timeout=10)
                    if resp.status_code == 200:
                        data = resp.content
                        fmt = "turtle" if url.endswith(".ttl") else "xml"
                        try:
                            base_graph.parse(data=data, format=fmt)
                        except Exception:
                            # If parsing fails with one format, try the other
                            alt_fmt = "xml" if fmt == "turtle" else "turtle"
                            try:
                                base_graph.parse(data=data, format=alt_fmt)
                            except Exception:
                                continue
                        # Successfully parsed base ontology
                        return base_graph
                except Exception:
                    continue
            # Return None if unable to fetch
            return None

        def build_extended_ontology(data: dict, base_graph: Graph = None):
            """
            Build the extended ontology as an rdflib Graph and a networkx MultiDiGraph.
            If base_graph is provided, use it as a starting point; otherwise, create base classes.
            """
            # Initialize rdflib graph
            graph = Graph()
            if base_graph:
                graph += base_graph  # include all triples from base ontology
            # Determine base namespace
            base_ns = None
            if base_graph:
                for subj in base_graph.subjects(RDF.type, OWL.Ontology):
                    base_ns = Namespace(str(subj) + "#")
                    break
            if not base_ns:
                base_ns = Namespace("http://example.org/matgraph#")
            # Bind common prefixes
            graph.bind("mat", base_ns)
            graph.bind("owl", OWL)
            graph.bind("rdfs", RDFS)
            graph.bind("rdf", RDF)
            # Core base classes (Material, Property, Parameter, Process)
            base_material = base_property = base_parameter = base_process = None
            if base_graph:
                # Try to find existing core classes in the base ontology
                for cls in base_graph.subjects(RDF.type, OWL.Class):
                    lname = cls.split("#")[-1].lower()
                    if lname in ("material", "matter"):
                        base_material = cls
                    elif lname == "property":
                        base_property = cls
                    elif lname == "parameter":
                        base_parameter = cls
                    elif lname == "process":
                        base_process = cls
            # If any core class not found, create new ones in base_ns
            if not base_material:
                base_material = base_ns.Material
                graph.add((base_material, RDF.type, OWL.Class))
            if not base_property:
                base_property = base_ns.Property
                graph.add((base_property, RDF.type, OWL.Class))
            if not base_parameter:
                base_parameter = base_ns.Parameter
                graph.add((base_parameter, RDF.type, OWL.Class))
            if not base_process:
                base_process = base_ns.Process
                graph.add((base_process, RDF.type, OWL.Class))
            # Add new ontology classes (subclasses for each category)
            for cat_key, parent_cls in [
                ("materials", base_material),
                ("properties", base_property),
                ("parameters", base_parameter),
                ("processes", base_process),
            ]:
                for name in data.get(cat_key, []):
                    class_uri = base_ns[to_valid_uri(name)]
                    graph.add((class_uri, RDF.type, OWL.Class))
                    graph.add((class_uri, RDFS.subClassOf, parent_cls))
            # Define object properties (hasProperty, influences), or reuse if present
            hasProperty = base_ns.hasProperty
            influences = base_ns.influences
            prop_exists = inf_exists = False
            if base_graph:
                for prop in base_graph.subjects(RDF.type, OWL.ObjectProperty):
                    lname = prop.split("#")[-1]
                    if lname == "hasProperty":
                        hasProperty = prop
                        prop_exists = True
                    if lname == "influences":
                        influences = prop
                        inf_exists = True
            if not prop_exists:
                graph.add((hasProperty, RDF.type, OWL.ObjectProperty))
                graph.add((hasProperty, RDFS.domain, base_material))
                graph.add((hasProperty, RDFS.range, base_property))
            if not inf_exists:
                graph.add((influences, RDF.type, OWL.ObjectProperty))
                graph.add((influences, RDFS.range, base_property))
            # Add relationships (triples) from the JSON data
            for rel in data.get("relationships", []):
                subj = rel.get("subject")
                pred = rel.get("predicate")
                obj = rel.get("object")
                if not (subj and pred and obj):
                    continue
                subj_uri = base_ns[to_valid_uri(subj)]
                obj_uri = base_ns[to_valid_uri(obj)]
                # Determine predicate URI
                if pred == "hasProperty":
                    pred_uri = hasProperty
                elif pred == "influences":
                    pred_uri = influences
                else:
                    pred_uri = base_ns[to_valid_uri(pred)]
                    graph.add((pred_uri, RDF.type, OWL.ObjectProperty))
                graph.add((subj_uri, pred_uri, obj_uri))
            # Create a networkx directed graph for visualization/GraphML
            G = nx.MultiDiGraph()
            # Add base class nodes (with category and flag)
            base_nodes = {
                str(base_material).split("#")[-1]: "Material",
                str(base_property).split("#")[-1]: "Property",
                str(base_parameter).split("#")[-1]: "Parameter",
                str(base_process).split("#")[-1]: "Process",
            }
            for node_label, category in base_nodes.items():
                G.add_node(node_label, category=category, is_base=1)
            # Add new class nodes and subclass-of edges
            for cat_key, parent_cls in [
                ("materials", base_material),
                ("properties", base_property),
                ("parameters", base_parameter),
                ("processes", base_process),
            ]:
                for name in data.get(cat_key, []):
                    class_label = to_valid_uri(name)
                    parent_label = parent_cls.split("#")[-1] if "#" in str(parent_cls) else str(parent_cls)
                    G.add_node(class_label, category=parent_label, is_base=0)
                    G.add_edge(class_label, parent_label, label="subClassOf")
            # Add relationship edges (hasProperty, influences, etc.)
            for rel in data.get("relationships", []):
                subj_label = to_valid_uri(rel.get("subject", ""))
                obj_label = to_valid_uri(rel.get("object", ""))
                pred_label = rel.get("predicate", "")
                if not subj_label or not obj_label or not pred_label:
                    continue
                if subj_label not in G:
                    G.add_node(subj_label, category="", is_base=0)
                if obj_label not in G:
                    G.add_node(obj_label, category="", is_base=0)
                G.add_edge(subj_label, obj_label, label=pred_label)
            return graph, G

        def generate_ontology_structure(question: str, features_list: list) -> dict:
            """Use GPT-4 (via CrewAI) to determine ontology classes and relationships for the input question/features."""
            # Ensure API key is set
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OpenAI API key not found. Please provide a valid API key.")
            # Configure LLMs for agents
            openai.api_key = os.getenv("OPENAI_API_KEY")
            llm_domain = LLM(model="openai/gpt-4", temperature=0.3, max_tokens=1000)
            llm_engineer = LLM(model="openai/gpt-4", temperature=0.0, max_tokens=1000)
            # Define the agents
            domain_agent = Agent(
                role="Domain Expert",
                goal="Identify relevant ontology classes from the question and features",
                backstory="An expert in materials and ontologies analyzing research questions.",
                llm=llm_domain
            )
            ontology_agent = Agent(
                role="Ontology Engineer",
                goal="Produce ontology extension in JSON format",
                backstory="An ontology engineer who converts domain knowledge into structured classes and properties.",
                llm=llm_engineer
            )
            # Craft prompts for each agent
            features_text = ", ".join(features_list)
            domain_prompt = (
                f"Analyze the research question: \"{question}\"\n"
                f"The relevant research features are: {features_text}.\n"
                "Determine which features correspond to a Material, a Property, a Parameter, or a Process in an ontology. "
                "Propose appropriate class names for each (use CamelCase for multi-word terms). "
                "Then suggest relationships between these classes to reflect the question. Use 'hasProperty' to link Materials to Properties, and 'influences' to link Parameters or Processes to Properties.\n\n"
                "Provide the output in a structured format with categories:\n"
                "- Material classes: ...\n"
                "- Property classes: ...\n"
                "- Parameter classes: ...\n"
                "- Process classes: ...\n"
                "- Relationships:\n"
                "  * MaterialX hasProperty PropertyY\n"
                "  * ParameterZ influences PropertyY\n"
            )
            ontology_prompt = (
                "You are the Ontology Engineer. Using the Domain Expert's analysis, produce a JSON object with the ontology extension.\n"
                "The JSON should have the structure:\n"
                "{\n"
                '  "materials": [...],\n'
                '  "properties": [...],\n'
                '  "parameters": [...],\n'
                '  "processes": [...],\n'
                '  "relationships": [\n'
                '    {"subject": "...", "predicate": "...", "object": "..."},\n'
                "    ...\n"
                "  ]\n"
                "}\n"
                "Include all keys (use empty lists [] if none). Use exactly 'hasProperty' and 'influences' for predicates. "
                "Do NOT add explanations, just output valid JSON."
            )
            # Create tasks for the agents
            domain_task = Task(
                description=domain_prompt,
                expected_output=(
                    "A structured list of ontology classes and relationships, categorized by Materials, Properties, Parameters, and Processes.\n"
                    "- Material classes: ...\n"
                    "- Property classes: ...\n"
                    "- Parameter classes: ...\n"
                    "- Process classes: ...\n"
                    "- Relationships:\n"
                    "  * MaterialX hasProperty PropertyY\n"
                    "  * ParameterZ influences PropertyY"
                ),
                agent=domain_agent
            )
            ontology_task = Task(
                description=ontology_prompt,
                expected_output=(
                    "A JSON object with keys 'materials', 'properties', 'parameters', 'processes', 'relationships'. "
                    "Each key's value is a list, and 'relationships' is a list of objects with 'subject', 'predicate', 'object'."
                ),
                agent=ontology_agent,
                context=[domain_task]
            )
            # Run the multi-agent Crew sequentially
            crew = Crew(
                agents=[domain_agent, ontology_agent],
                tasks=[domain_task, ontology_task],
                process=Process.sequential
            )
            crew.kickoff()  # execute the tasks sequentially
            # Extract the JSON output from the ontology task
            ont_output = ontology_task.output
            if hasattr(ont_output, "json_dict") and ont_output.json_dict:
                data = ont_output.json_dict
            else:
                raw_text = str(ont_output).strip()
                if raw_text.startswith("```"):
                    # Remove markdown formatting if present
                    raw_text = raw_text.strip("```").strip()
                data = json.loads(raw_text)
            return data

        # Run the ontology generation process
        try:
            ontology_data = generate_ontology_structure(question, features)
        except Exception as e:
            st.error(f"Ontology generation failed: {e}")
            st.stop()
    # Ontology JSON structure obtained successfully

    # Build the extended ontology graphs (rdflib and networkx)
    with st.spinner("Building ontology graph..."):
        try:
            base_graph = fetch_base_ontology()
        except Exception as e:
            base_graph = None
        if base_graph is None:
            st.warning("Base ontology not found. Proceeding with minimal base classes.")
        try:
            extended_graph, graphml_graph = build_extended_ontology(ontology_data, base_graph)
        except Exception as e:
            st.error(f"Error building ontology graph: {e}")
            st.stop()

    st.success("Ontology generation complete! ?")

    # Prepare Turtle serialization for download
    try:
        turtle_data = extended_graph.serialize(format="turtle")
    except Exception as e:
        turtle_data = None
        st.error(f"Failed to serialize ontology to Turtle format: {e}")
    if turtle_data:
        st.download_button(
            label="Download Ontology (Turtle .ttl)",
            data=turtle_data,
            file_name="extended_ontology.ttl",
            mime="text/turtle"
        )

    # Prepare GraphML serialization for download
    import io
    try:
        buffer = io.BytesIO()
        nx.write_graphml(graphml_graph, buffer)
        graphml_data = buffer.getvalue().decode("utf-8")
    except Exception as e:
        graphml_data = None
        st.error(f"Failed to generate GraphML data: {e}")
    if graphml_data:
        st.download_button(
            label="Download Ontology Graph (GraphML .graphml)",
            data=graphml_data,
            file_name="extended_ontology.graphml",
            mime="application/xml"
        )

    # Visualize the ontology graph using Graphviz
    try:
        from graphviz import Digraph
        dot = Digraph(name="OntologyGraph", format="png")
        dot.attr(rankdir="TB")  # Top-to-bottom layout
        # Add nodes (base classes in gray for clarity)
        for node, attrs in graphml_graph.nodes(data=True):
            label = str(node)
            if attrs.get("is_base") == 1:
                dot.node(label, label=label, style="filled", fillcolor="lightgray", shape="ellipse")
            else:
                dot.node(label, label=label, shape="ellipse")
        # Add edges with labels
        for u, v, edge_attrs in graphml_graph.edges(data=True):
            edge_label = edge_attrs.get("label", "")
            dot.edge(str(u), str(v), label=edge_label)
        st.subheader("Ontology Graph Visualization")
        st.graphviz_chart(dot.source)
    except Exception as e:
        st.error(f"Failed to display graph visualization: {e}")
