#!/usr/bin/env python3
"""
Inference API Server for trained models.

This server provides a REST API for text generation using trained models.

Usage:
    # Start server
    python -m experiment.server --checkpoint step_002920 --port 8000
    
    # Start with specific host
    python -m experiment.server --checkpoint step_002920 --host 0.0.0.0 --port 8000

API Endpoints:
    POST /generate - Generate text from prompt
    GET /health - Health check
    GET /info - Model information
"""

import os
import sys
import argparse
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment.inference import InferenceEngine

# Global inference engine and model info
inference_engine = None
model_info = {
    'checkpoint': os.getenv('INFERENCE_CHECKPOINT'),
    'device': os.getenv('INFERENCE_DEVICE'),
}


class GenerateRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., description="Input prompt text", min_length=1)
    max_tokens: int = Field(100, description="Maximum tokens to generate", ge=1, le=2048)
    temperature: float = Field(0.8, description="Sampling temperature", ge=0.0, le=2.0)
    top_k: Optional[int] = Field(None, description="Top-k sampling (None = use all tokens)", ge=1, le=32)
    stop_on_newline: bool = Field(False, description="Stop generation at newline")


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    prompt: str = Field(..., description="Input prompt")
    generated_text: str = Field(..., description="Generated text (including prompt)")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    model: str = Field(..., description="Model checkpoint used")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    checkpoint: str = Field(..., description="Loaded checkpoint name")


class InfoResponse(BaseModel):
    """Response model for model information."""
    checkpoint: str = Field(..., description="Checkpoint name")
    step: int = Field(..., description="Training step")
    val_loss: float = Field(..., description="Validation loss at checkpoint")
    model_layers: int = Field(..., description="Number of model layers")
    model_dim: int = Field(..., description="Model dimension")
    vocab_size: int = Field(..., description="Vocabulary size")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    checkpoint = model_info.get('checkpoint')
    if checkpoint:
        print(f"Loading model from checkpoint: {checkpoint}")
        global inference_engine
        try:
            inference_engine = InferenceEngine.from_checkpoint(
                checkpoint,
                device=model_info.get('device'),
                temperature=0.8,
                top_k=None
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Warning: No checkpoint specified")
    yield
    # Shutdown
    print("Shutting down server...")


# Create FastAPI app
app = FastAPI(
    title="Inference API",
    description="REST API for text generation using trained models",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate text from a prompt.
    
    Parameters:
    - **prompt**: Input text to continue from
    - **max_tokens**: Maximum number of tokens to generate (1-2048)
    - **temperature**: Sampling temperature (0.0=greedy, 2.0=very random)
    - **top_k**: Top-k sampling, only use top k tokens (None=use all)
    - **stop_on_newline**: Stop generation when newline is encountered
    
    Returns generated text including the prompt.
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get prompt length before generation
        prompt_tokens = inference_engine.tokenizer.encode(request.prompt)
        prompt_len = len(prompt_tokens)
        
        # Generate
        generated_text = inference_engine.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            stop_on_newline=request.stop_on_newline
        )
        
        # Calculate tokens generated
        total_tokens = len(inference_engine.tokenizer.encode(generated_text))
        tokens_generated = total_tokens - prompt_len
        
        return GenerateResponse(
            prompt=request.prompt,
            generated_text=generated_text,
            tokens_generated=tokens_generated,
            model=model_info['checkpoint']
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.
    
    Returns service status and model information.
    """
    return HealthResponse(
        status="ok" if inference_engine is not None else "error",
        model_loaded=inference_engine is not None,
        checkpoint=model_info.get('checkpoint', 'unknown')
    )


@app.get("/info", response_model=InfoResponse)
async def info():
    """
    Get model information.
    
    Returns details about the loaded model checkpoint.
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Get model config from the loaded model
    model = inference_engine.model
    
    return InfoResponse(
        checkpoint=model_info['checkpoint'],
        step=model_info.get('step', 0),
        val_loss=model_info.get('val_loss', 0.0),
        model_layers=model.config.n_layer,
        model_dim=model.config.n_embd,
        vocab_size=model.config.vocab_size
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Inference API",
        "version": "1.0.0",
        "checkpoint": model_info.get('checkpoint', 'unknown'),
        "status": "running" if inference_engine is not None else "loading",
        "endpoints": {
            "POST /generate": "Generate text from prompt",
            "GET /health": "Health check",
            "GET /info": "Model information",
            "GET /docs": "Interactive API documentation"
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Start inference API server")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint folder name (e.g., step_002920)")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to bind to (default: 8000)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cpu/cuda/mps, default=auto)")
    parser.add_argument("--reload", action="store_true",
                        help="Enable auto-reload on code changes (dev mode)")
    
    args = parser.parse_args()
    
    # Set environment variables for the model config
    os.environ['INFERENCE_CHECKPOINT'] = args.checkpoint
    if args.device:
        os.environ['INFERENCE_DEVICE'] = args.device
    
    # Store model info globally
    global model_info
    model_info['checkpoint'] = args.checkpoint
    model_info['device'] = args.device
    
    # Get checkpoint info if available
    try:
        import json
        script_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_dir = os.path.join(script_dir, "checkpoints", args.checkpoint)
        
        # Find meta file
        import glob
        meta_files = glob.glob(os.path.join(checkpoint_dir, "meta_*.json"))
        if meta_files:
            with open(meta_files[0], 'r') as f:
                meta = json.load(f)
                model_info['step'] = meta.get('step', 0)
                model_info['val_loss'] = meta.get('val_loss', 0.0)
    except:
        pass
    
    print(f"\n{'='*60}")
    print(f"Starting Inference API Server")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Device: {args.device or 'auto-detect'}")
    print(f"{'='*60}\n")
    print(f"API Documentation: http://{args.host}:{args.port}/docs")
    print(f"Health Check: http://{args.host}:{args.port}/health")
    print(f"{'='*60}\n")
    
    # Start server
    uvicorn.run(
        "experiment.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
