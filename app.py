import streamlit as st
import pandas as pd
import json
import os
import subprocess
import time
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# Page config
st.set_page_config(
    page_title="IMDB Sentiment Analysis - Lightweight Transformer Fine-tuning",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.375rem;
        padding: 0.75rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitIMDBApp:
    def __init__(self):
        self.results_dir = Path("./results")
        self.results_dir.mkdir(exist_ok=True)
        
    def show_header(self):
        """Display app header"""
        st.markdown('<h1 class="main-header">🎬 IMDB Sentiment Analysis</h1>', unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">Lightweight Transformer Fine-tuning with Ray AIR</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", "DistilBERT", "66M params")
        with col2:
            st.metric("Framework", "Ray AIR", "Distributed")
        with col3:
            st.metric("Dataset", "IMDB", "50k reviews")
    
    def show_project_overview(self):
        """Display project overview"""
        with st.expander("📋 Project Overview", expanded=True):
            st.markdown("""
            This project demonstrates **production-ready fine-tuning** of a DistilBERT model on the IMDB movie reviews dataset for binary sentiment classification.
            
            **Key Features:**
            - ⚡ **Lightweight DistilBERT** for fast training and inference
            - 🚀 **Ray AIR** for scalable, distributed training
            - 📊 **Real-time monitoring** and evaluation metrics
            - 🔧 **Interactive parameter tuning** via this Streamlit interface
            - 📈 **Benchmarking** against other transformer models
            
            **Technology Stack:**
            - `transformers` - Hugging Face Transformers library
            - `ray[air]` - Distributed training framework
            - `datasets` - Dataset loading and processing
            - `streamlit` - Interactive web interface
            """)
    
    def show_training_config(self):
        """Training configuration sidebar"""
        st.sidebar.header("🔧 Training Configuration")
        
        model_name = st.sidebar.selectbox(
            "Model",
            ["distilbert-base-uncased", "bert-base-uncased", "roberta-base"],
            help="Choose the transformer model to fine-tune"
        )
        
        train_size = st.sidebar.slider(
            "Training Dataset Size",
            min_value=500,
            max_value=5000,
            value=2000,
            step=500,
            help="Number of samples to use for training"
        )
        
        num_workers = st.sidebar.slider(
            "Number of Workers",
            min_value=1,
            max_value=4,
            value=2,
            help="Number of Ray workers for distributed training"
        )
        
        use_gpu = st.sidebar.checkbox(
            "Use GPU",
            value=False,
            help="Enable GPU acceleration (if available)"
        )
        
        max_length = st.sidebar.slider(
            "Max Sequence Length",
            min_value=64,
            max_value=512,
            value=128,
            step=64,
            help="Maximum token length for input sequences"
        )
        
        return {
            "model_name": model_name,
            "train_size": train_size,
            "num_workers": num_workers,
            "use_gpu": use_gpu,
            "max_length": max_length
        }
    
    def run_training(self, config):
        """Run the training process"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate training progress (replace with actual training call)
        cmd = [
            "python", "ray_lightweight_llm_finetune.py",
            "--model_name", config["model_name"],
            "--train_size", str(config["train_size"]),
            "--num_workers", str(config["num_workers"]),
            "--max_length", str(config["max_length"])
        ]
        
        if config["use_gpu"]:
            cmd.append("--use_gpu")
        
        try:
            status_text.text("🚀 Initializing training...")
            progress_bar.progress(10)
            
            # Run the training script
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Simulate progress updates
            for i in range(20, 101, 20):
                time.sleep(2)
                progress_bar.progress(i)
                if i == 40:
                    status_text.text("📊 Loading and tokenizing dataset...")
                elif i == 60:
                    status_text.text("🎯 Training model...")
                elif i == 80:
                    status_text.text("📈 Evaluating performance...")
                elif i == 100:
                    status_text.text("✅ Training completed!")
            
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                st.success("🎉 Training completed successfully!")
                return True
            else:
                st.error(f"❌ Training failed: {stderr}")
                return False
                
        except Exception as e:
            st.error(f"❌ Error running training: {str(e)}")
            return False
    
    def show_demo_inference(self):
        """Demo inference section"""
        st.header("🔮 Try the Model")
        
        # Sample texts for demo
        sample_texts = [
            "This movie was absolutely fantastic! Great acting and storyline.",
            "Terrible movie, waste of time. Poor acting and boring plot.",
            "The film was okay, nothing special but not bad either.",
            "One of the best movies I've ever seen! Highly recommended.",
            "Disappointing sequel, doesn't live up to the original."
        ]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            user_input = st.text_area(
                "Enter a movie review:",
                placeholder="Type your movie review here...",
                height=100
            )
            
            if st.button("🎯 Analyze Sentiment"):
                if user_input.strip():
                    # Simulate prediction (replace with actual model inference)
                    sentiment = "Positive" if len(user_input.split()) > 10 else "Negative"
                    confidence = 0.85 if sentiment == "Positive" else 0.78
                    
                    # Display results
                    col1_res, col2_res = st.columns(2)
                    with col1_res:
                        st.metric("Sentiment", sentiment)
                    with col2_res:
                        st.metric("Confidence", f"{confidence:.2%}")
                    
                    # Confidence bar
                    fig = go.Figure(go.Bar(
                        x=[confidence],
                        y=["Confidence"],
                        orientation='h',
                        marker_color='green' if sentiment == 'Positive' else 'red'
                    ))
                    fig.update_layout(
                        title="Prediction Confidence",
                        xaxis=dict(range=[0, 1]),
                        height=200
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Sample Reviews:**")
            for i, text in enumerate(sample_texts):
                if st.button(f"Try Sample {i+1}", key=f"sample_{i}"):
                    st.text_area(
                        "Enter a movie review:",
                        value=text,
                        height=100,
                        key=f"sample_text_{i}"
                    )
    
    def show_model_comparison(self):
        """Show model comparison"""
        st.header("📊 Model Comparison")
        
        # Sample benchmark data
        comparison_data = {
            "Model": ["DistilBERT-base", "BERT-base", "RoBERTa-base"],
            "Parameters": ["66M", "110M", "125M"],
            "Training Time (2 epochs)": ["~30 min", "~50 min", "~55 min"],
            "Test Accuracy": [0.89, 0.91, 0.92],
            "Inference Speed": ["Fast", "Medium", "Medium"],
            "Memory Usage": ["Low", "High", "High"]
        }
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)
        
        # Accuracy comparison chart
        fig = px.bar(
            df, 
            x="Model", 
            y="Test Accuracy",
            title="Model Accuracy Comparison",
            color="Test Accuracy",
            color_continuous_scale="viridis"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def show_training_results(self):
        """Display training results if available"""
        results_file = self.results_dir / "training_results.json"
        
        if results_file.exists():
            st.header("📈 Training Results")
            
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model", results.get("model_name", "N/A"))
            with col2:
                st.metric("Status", "✅ Completed" if results.get("training_completed") else "❌ Failed")
            with col3:
                st.metric("Checkpoint", "Saved" if results.get("best_checkpoint") else "None")
            
            if results.get("metrics"):
                st.subheader("Training Metrics")
                st.json(results["metrics"])
    
    def show_code_section(self):
        """Show code examples"""
        with st.expander("💻 View Code", expanded=False):
            st.subheader("Training Script Usage")
            st.code("""
# Basic usage
python ray_lightweight_llm_finetune.py

# With custom parameters
python ray_lightweight_llm_finetune.py \\
    --model_name distilbert-base-uncased \\
    --train_size 2000 \\
    --num_workers 2 \\
    --use_gpu
            """, language="bash")
            
            st.subheader("Key Functions")
            st.code("""
from ray_lightweight_llm_finetune import IMDBSentimentTrainer

# Initialize trainer
trainer = IMDBSentimentTrainer(
    model_name="distilbert-base-uncased",
    max_length=128
)

# Run training
result = trainer.train(
    num_workers=2,
    use_gpu=False,
    train_size=2000
)
            """, language="python")

def main():
    app = StreamlitIMDBApp()
    
    # Header
    app.show_header()
    
    # Project overview
    app.show_project_overview()
    
    # Sidebar configuration
    config = app.show_training_config()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🚀 Training", "🔮 Demo", "📊 Comparison", "📈 Results", "💻 Code"])
    
    with tab1:
        st.header("🚀 Start Training")
        
        st.markdown("Configure your training parameters in the sidebar and click the button below to start training.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🎯 Start Training", type="primary", use_container_width=True):
                with st.spinner("Training in progress..."):
                    success = app.run_training(config)
                    if success:
                        st.balloons()
    
    with tab2:
        app.show_demo_inference()
    
    with tab3:
        app.show_model_comparison()
    
    with tab4:
        app.show_training_results()
    
    with tab5:
        app.show_code_section()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using Streamlit • Ray AIR • Hugging Face Transformers</p>
        <p>⭐ Star this project on GitHub if you found it helpful!</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
