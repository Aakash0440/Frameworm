"""
Complete end-to-end workflow test.

Tests the entire FRAMEWORM pipeline from data to deployment.
"""

import pytest
import torch
import tempfile
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset


class TestCompleteWorkflow:
    """Test complete training → export → deployment workflow"""
    
    def test_full_pipeline(self):
        """Complete pipeline: train → track → search → export → serve"""
        
        from core import Config, get_model, Trainer
        from training.callbacks import EarlyStopping, ModelCheckpoint
        from experiment import Experiment
        from search import RandomSearch
        from deployment import ModelExporter
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # 1. TRAINING
            print("\n1️⃣  Testing training...")
            config = Config.from_dict({
                'model': {'type': 'vae', 'in_channels': 3, 'latent_dim': 16}
            })
            
            model = get_model('vae')(config)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            trainer = Trainer(model, optimizer, device='cpu')
            
            # Synthetic data
            train_data = TensorDataset(torch.randn(100, 3, 32, 32))
            val_data = TensorDataset(torch.randn(20, 3, 32, 32))
            train_loader = DataLoader(train_data, batch_size=16)
            val_loader = DataLoader(val_data, batch_size=16)
            
            # Add callbacks
            checkpoint_path = tmpdir / 'checkpoint.pt'
            trainer.add_callback(ModelCheckpoint(str(checkpoint_path), save_best_only=True))
            trainer.add_callback(EarlyStopping(patience=2))
            
            # Train with experiment tracking
            with Experiment(name='test-exp', base_dir=str(tmpdir)) as exp:
                trainer.set_experiment(exp)
                trainer.train(train_loader, val_loader, epochs=5)
                
                assert checkpoint_path.exists(), "Checkpoint should be saved"
                assert exp.experiment_id is not None
                print("   ✅ Training complete")
            
            # 2. HYPERPARAMETER SEARCH
            print("\n2️⃣  Testing hyperparameter search...")
            
            def train_fn(config):
                model = get_model('vae')(config)
                optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
                trainer = Trainer(model, optimizer, device='cpu')
                trainer.train(train_loader, val_loader, epochs=2)
                return {'val_loss': trainer.state.val_metrics['loss'][-1]}
            
            search_config = config.copy()
            search_config.training = Config.from_dict({'lr': 0.001})
            
            search = RandomSearch(
                base_config=search_config,
                search_space={'training.lr': [0.0001, 0.001, 0.01]},
                metric='val_loss',
                mode='min',
                num_trials=3
            )
            
            best_config, best_score = search.run(train_fn)
            assert best_config is not None
            assert best_score > 0
            print("   ✅ Hyperparameter search complete")
            
            # 3. MODEL EXPORT
            print("\n3️⃣  Testing model export...")
            
            model = get_model('vae')(config)
            model.load_state_dict(torch.load(checkpoint_path))
            
            exporter = ModelExporter(model)
            
            # TorchScript
            ts_path = tmpdir / 'model.pt'
            exporter.to_torchscript(str(ts_path))
            assert ts_path.exists()
            
            # Load and test
            loaded = torch.jit.load(str(ts_path))
            test_input = torch.randn(1, 3, 32, 32)
            output = loaded(test_input)
            assert output['recon'].shape == test_input.shape
            print("   ✅ Model export complete")
            
            # 4. DEPLOYMENT SERVER (mock test)
            print("\n4️⃣  Testing deployment components...")
            from deployment.server import create_app
            
            app = create_app(str(ts_path))
            assert app is not None
            print("   ✅ Deployment ready")
            
            # 5. METRICS
            print("\n5️⃣  Testing metrics...")
            from metrics import FID
            
            fid = FID(device='cpu')
            real_images = torch.randn(10, 3, 64, 64)
            fake_images = torch.randn(10, 3, 64, 64)
            score = fid(real_images, fake_images)
            assert score >= 0
            print("   ✅ Metrics working")
            
            print("\n✅ COMPLETE PIPELINE TESTED SUCCESSFULLY!\n")
    
    def test_all_models(self):
        """Test that all registered models work"""
        
        from core import get_model, Config
        
        models = ['vae', 'dcgan', 'ddpm', 'vqvae2', 'vitgan', 'cfg_ddpm']
        
        print("\n6️⃣  Testing all models...")
        for model_name in models:
            config = Config.from_dict({
                'model': {
                    'type': model_name,
                    'in_channels': 3,
                    'latent_dim': 16,
                    'image_size': 32,
                    'image_channels': 3,
                    'hidden_channels': 32,
                    'num_embeddings': 64,
                    'embedding_dim': 16,
                    'num_classes': 5,
                    'model_channels': 32,
                    'timesteps': 10
                }
            })
            
            model = get_model(model_name)(config)
            x = torch.randn(2, 3, 32, 32)
            
            if model_name == 'cfg_ddpm':
                output = model.compute_loss(x, torch.randint(0, 5, (2,)))
            else:
                output = model.compute_loss(x)
            
            assert 'loss' in output
            print(f"   ✅ {model_name}")
        
        print("   ✅ All models working!\n")


# Run tests
if __name__ == '__main__':
    test = TestCompleteWorkflow()
    test.test_full_pipeline()
    test.test_all_models()
    print("🎉 ALL TESTS PASSED!")