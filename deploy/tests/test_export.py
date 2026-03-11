"""Tests for model_exporter and server_builder."""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, ".")


def test_server_builder_generates_file():
    from deploy.core.server_builder import ServerBuilder

    builder = ServerBuilder()
    with tempfile.TemporaryDirectory() as d:
        path = builder.build(
            model_type="dcgan",
            model_name="test_gan",
            model_version="v1.0",
            model_path="model/test.pt",
            output_dir=d,
            port=8001,
        )
        assert os.path.exists(path)
        code = open(path).read()
        assert "test_gan" in code
        assert "v1.0" in code
        assert "InferenceRequest" in code
        assert "build_app" in code


def test_server_builder_all_model_types():
    from deploy.core.server_builder import MODEL_SPECS, ServerBuilder

    builder = ServerBuilder()
    with tempfile.TemporaryDirectory() as d:
        for model_type in MODEL_SPECS:
            out = os.path.join(d, model_type)
            path = builder.build(
                model_type=model_type,
                model_name=f"test_{model_type}",
                model_version="v0.1",
                model_path="model/m.pt",
                output_dir=out,
            )
            assert os.path.exists(path), f"Missing server for {model_type}"


def test_server_builder_shift_injection():
    from deploy.core.server_builder import ServerBuilder

    builder = ServerBuilder()
    with tempfile.TemporaryDirectory() as d:
        path = builder.build(
            model_type="vae",
            model_name="vae_test",
            model_version="v1",
            model_path="m.pt",
            output_dir=d,
            shift_reference="my_reference",
        )
        code = open(path).read()
        assert "my_reference" in code


def test_dockerfile_generated():
    from deploy.core.docker_builder import DockerBuilder

    builder = DockerBuilder()
    with tempfile.TemporaryDirectory() as d:
        path = builder.generate_dockerfile(
            server_dir=d,
            model_name="test",
            model_version="v1",
            model_path="model/test.pt",
            port=8080,
        )
        assert os.path.exists(path)
        content = open(path).read()
        assert "HEALTHCHECK" in content
        assert "frameworm" in content  # non-root user
        assert "EXPOSE 8080" in content


def test_docker_compose_generated():
    from deploy.core.docker_builder import DockerBuilder

    with tempfile.TemporaryDirectory() as d:
        path = DockerBuilder().generate_compose(
            model_name="mymodel",
            model_version="v2",
            port=9000,
            output_path=os.path.join(d, "docker-compose.yml"),
        )
        assert os.path.exists(path)
        content = open(path).read()
        assert "mymodel" in content
        assert "9000:9000" in content
