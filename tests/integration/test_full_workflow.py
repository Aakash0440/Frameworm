import pytest
from core.config import Config
from graph.graph import Graph, Node
from models.gan.dcgan import DCGAN
from models.vae.vanilla import VAE


class TestCompleteWorkflow:

    def test_model_pipeline_integration(self):
        """Integrate a GAN model in a simple pipeline"""

        # Create graph
        graph = Graph()

        # Node functions
        def create_model():
            cfg = Config("configs/models/gan/dcgan.yaml")
            return DCGAN(cfg)

        def generate(model):
            return model(batch_size=2)

        def process(images):
            return images.mean()

        # Add nodes
        graph.add_node(Node("model", create_model))
        graph.add_node(Node("generate", generate, depends_on=["model"]))
        graph.add_node(Node("process", process, depends_on=["generate"]))

        # Execute graph
        results = graph.execute()

        # Validate output
        assert "process" in results
        assert isinstance(results["process"].item(), float)

    def test_parallel_model_inference(self):
        """Run parallel inference on multiple models"""

        # Create graph
        graph = Graph()

        # Load models
        def load_dcgan():
            cfg = Config("configs/models/gan/dcgan.yaml")
            return DCGAN(cfg)

        def load_vae():
            cfg = Config("configs/models/vae/vanilla.yaml")
            # Patch removed call to init_weights if needed
            # VAE will initialize weights internally
            return VAE(cfg)

        # Generate outputs
        def gen_dcgan(model):
            return model(batch_size=1)

        def gen_vae(model):
            return model.sample(1)

        # Compare outputs
        def compare(dcgan_out, vae_out):
            return {"dcgan_mean": dcgan_out.mean().item(), "vae_mean": vae_out.mean().item()}

        # Add nodes
        graph.add_node(Node("dcgan", load_dcgan))
        graph.add_node(Node("vae", load_vae))
        graph.add_node(Node("gen_dcgan", gen_dcgan, depends_on=["dcgan"]))
        graph.add_node(Node("gen_vae", gen_vae, depends_on=["vae"]))
        graph.add_node(Node("compare", compare, depends_on=["gen_dcgan", "gen_vae"]))

        # Execute in parallel
        results = graph.execute_parallel(max_workers=2)

        # Validate results
        assert "compare" in results
        assert isinstance(results["compare"]["dcgan_mean"], float)
        assert isinstance(results["compare"]["vae_mean"], float)
