"""
Quick Start Training Script for Deepfake Detector
Run this to train the model on your processed dataset
"""

import sys
sys.path.append('src')

import torch
from pathlib import Path
import argparse

from models.deepfake_detector import DeepfakeDetector
from training.trainer import DeepfakeTrainer
from training.dataset_loader import create_dataloaders
from evaluation.evaluator import DeepfakeEvaluator


def main(args):
    """Main training function"""
    
    print("\n" + "="*70)
    print("DEEPFAKE DETECTOR - TRAINING")
    print("="*70 + "\n")
    
    # Check if processed data exists
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print("\nPlease run preprocessing first:")
        print("  cd src/preprocessing")
        print("  python process_dataset.py --num-videos None  # Process all videos")
        return
    
    print(f"✅ Found processed data: {data_path}\n")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"🖥️  Using device: {device}\n")
    
    # Create dataloaders
    print("📦 Loading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=str(data_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_handcrafted=args.use_handcrafted,
        sequence_length=args.sequence_length,
        augment_train=args.augment
    )
    
    # Initialize model
    print("🏗️  Building model...")
    model = DeepfakeDetector(
        sequence_length=args.sequence_length,
        input_channels=3,
        num_regions=7,
        temporal_feature_dim=args.feature_dim,
        transformer_dim=args.feature_dim,
        num_transformer_layers=args.num_layers,
        num_heads=args.num_heads,
        fusion_hidden_dim=args.fusion_dim,
        handcrafted_dim=49 if args.use_handcrafted else 0,
        use_handcrafted=args.use_handcrafted,
        dropout=args.dropout,
        num_classes=2
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    print(f"   Model size: ~{total_params * 4 / (1024**2):.2f} MB\n")
    
    # Initialize trainer
    print("🚀 Initializing trainer...")
    trainer = DeepfakeTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_focal_loss=args.use_focal_loss,
        max_epochs=args.epochs,
        grad_clip=args.grad_clip,
        use_amp=args.use_amp and torch.cuda.is_available(),
        use_scheduler=args.use_scheduler,
        early_stop_patience=args.early_stop_patience,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )
    
    # Train
    history = trainer.train()
    
    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    trainer.model.eval()
    
    evaluator = DeepfakeEvaluator(trainer.model, device=device)
    test_results = evaluator.evaluate(test_loader, return_predictions=True)
    
    # Print summary
    evaluator.print_summary(test_results['metrics'])
    
    # Generate visualizations
    if args.visualize:
        print("\n📈 Generating visualizations...")
        evaluator.visualize_results(test_results, output_dir=args.output_dir)
        evaluator.save_report(test_results, f"{args.output_dir}/test_evaluation.json")
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📁 Outputs saved to:")
    print(f"   Checkpoints: {args.checkpoint_dir}")
    print(f"   Logs: {args.log_dir}")
    print(f"   Evaluation: {args.output_dir}")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Deepfake Detector")
    
    # Data
    parser.add_argument('--data-path', type=str,
                       default='outputs/processed_features/dataset_features_chrom.pkl',
                       help='Path to processed features')
    parser.add_argument('--sequence-length', type=int, default=150,
                       help='Sequence length (frames)')
    parser.add_argument('--use-handcrafted', action='store_true', default=True,
                       help='Use hand-crafted features')
    parser.add_argument('--augment', action='store_true', default=True,
                       help='Augment training data')
    
    # Model
    parser.add_argument('--feature-dim', type=int, default=256,
                       help='Feature dimension')
    parser.add_argument('--num-layers', type=int, default=4,
                       help='Number of transformer layers')
    parser.add_argument('--num-heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--fusion-dim', type=int, default=512,
                       help='Fusion hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                       help='Gradient clipping')
    
    # Loss & optimization
    parser.add_argument('--use-focal-loss', action='store_true', default=True,
                       help='Use focal loss')
    parser.add_argument('--use-scheduler', action='store_true', default=True,
                       help='Use learning rate scheduler')
    parser.add_argument('--early-stop-patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--use-amp', action='store_true', default=True,
                       help='Use automatic mixed precision')
    
    # Hardware
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Output
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Log directory')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Evaluation output directory')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Generate visualizations')
    
    args = parser.parse_args()
    
    main(args)

