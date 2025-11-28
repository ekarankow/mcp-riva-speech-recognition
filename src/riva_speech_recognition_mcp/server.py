#!/usr/bin/env python3
"""
Riva Speech Recognition MCP Server

A Model Context Protocol server that performs speech recognition using NVIDIA Riva ASR
on properly formatted audio data.
"""

import argparse
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
import traceback
import logging
import sys
import base64
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
from pydantic_settings import BaseSettings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('riva_speech_recognition.log')
    ]
)
logger = logging.getLogger(__name__)

# Create the MCP server instance
mcp = FastMCP("Riva Speech Recognition MCP Server")


class RivaConfig(BaseSettings):
    """Configuration for NVIDIA Riva ASR settings."""
    
    # Server settings
    host: str = "localhost"
    port: int = 8080
    log_level: str = "INFO"
    
    # NVIDIA Riva server configuration
    riva_uri: str = os.getenv('RIVA_URI', 'localhost:50051')
    riva_language_code: str = "en-US"
    riva_asr_mode: str = os.getenv('RIVA_ASR_MODE', 'offline').lower()  # offline or streaming
    riva_max_alternatives: int = int(os.getenv('RIVA_MAX_ALTERNATIVES', '3'))
    riva_enable_punctuation: bool = True
    riva_verbatim_transcripts: bool = False
    
    class Config:
        env_file = ".env"
        env_prefix = "RIVA_"


# Global configuration
config = RivaConfig()


class TranscriptionAlternative(BaseModel):
    """Single transcription alternative with confidence score."""
    transcript: str
    confidence: float


class TranscriptionSegment(BaseModel):
    """Transcription segment with multiple alternatives."""
    alternatives: List[TranscriptionAlternative]
    best_transcript: str
    best_confidence: float


class TranscriptionResponse(BaseModel):
    """Response model for speech recognition."""
    success: bool
    transcript: str = ""
    confidence: float = 0.0
    segments: List[TranscriptionSegment] = []
    processing_mode: str = ""
    audio_duration_ms: float = 0.0
    processing_time_ms: float = 0.0
    error_message: str = ""


def transcribe_with_riva_offline(asr_service, config_obj, audio_data: bytes) -> TranscriptionResponse:
    """
    Perform offline speech recognition using NVIDIA Riva.
    
    Args:
        asr_service: Initialized Riva ASR service
        config_obj: Riva recognition config
        audio_data: Raw audio data bytes
        
    Returns:
        TranscriptionResponse: Transcription results
    """
    try:
        import time
        start_time = time.time()
        
        logger.info("Performing offline speech recognition...")
        response = asr_service.offline_recognize(audio_data, config_obj)
        
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"Offline recognition completed in {processing_time:.2f}ms")
        
        if not response.results:
            logger.warning("No results returned from offline recognition")
            return TranscriptionResponse(
                success=True,
                transcript="",
                confidence=0.0,
                processing_mode="offline",
                processing_time_ms=processing_time,
                error_message="No speech detected in audio"
            )
        
        logger.info(f"Processing {len(response.results)} result segments")
        
        # Process all segments
        segments = []
        transcript_parts = []
        confidence_scores = []
        
        for i, result in enumerate(response.results):
            if result.alternatives:
                # Convert alternatives to our model
                alternatives = [
                    TranscriptionAlternative(
                        transcript=alt.transcript,
                        confidence=alt.confidence
                    )
                    for alt in result.alternatives
                ]
                
                # Find best alternative
                best_alt = max(result.alternatives, key=lambda alt: alt.confidence)
                
                segment = TranscriptionSegment(
                    alternatives=alternatives,
                    best_transcript=best_alt.transcript,
                    best_confidence=best_alt.confidence
                )
                
                segments.append(segment)
                transcript_parts.append(best_alt.transcript)
                confidence_scores.append(best_alt.confidence)
                
                logger.debug(f"Segment {i+1}: '{best_alt.transcript}' (confidence: {best_alt.confidence:.3f})")
            else:
                logger.warning(f"Segment {i+1} has no alternatives")
        
        # Combine results
        final_transcript = " ".join(transcript_parts).strip()
        
        # Calculate weighted confidence
        if confidence_scores:
            if transcript_parts:
                segment_lengths = [len(part) for part in transcript_parts]
                total_length = sum(segment_lengths)
                
                if total_length > 0:
                    weighted_confidence = sum(
                        conf * length / total_length 
                        for conf, length in zip(confidence_scores, segment_lengths)
                    )
                else:
                    weighted_confidence = sum(confidence_scores) / len(confidence_scores)
            else:
                weighted_confidence = sum(confidence_scores) / len(confidence_scores)
        else:
            weighted_confidence = 0.0
        
        logger.info(f"Final transcript: '{final_transcript}' (confidence: {weighted_confidence:.3f})")
        
        return TranscriptionResponse(
            success=True,
            transcript=final_transcript,
            confidence=weighted_confidence,
            segments=segments,
            processing_mode="offline",
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        error_msg = f"Offline speech recognition failed: {e}"
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return TranscriptionResponse(
            success=False,
            error_message=error_msg,
            processing_mode="offline"
        )


def transcribe_with_riva_streaming(asr_service, config_obj, audio_data: bytes) -> TranscriptionResponse:
    """
    Perform streaming speech recognition using NVIDIA Riva.
    
    Args:
        asr_service: Initialized Riva ASR service
        config_obj: Riva recognition config
        audio_data: Raw audio data bytes
        
    Returns:
        TranscriptionResponse: Transcription results
    """
    try:
        import time
        import riva.client
        
        start_time = time.time()
        
        logger.info("Performing streaming speech recognition...")
        
        # Configure streaming settings
        streaming_config = riva.client.StreamingRecognitionConfig(
            config=config_obj,
            interim_results=True
        )
        
        # Create audio chunks generator
        def audio_chunks_generator():
            chunk_size = 1024 * 16  # 16KB chunks
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                if chunk:
                    logger.debug(f"Yielding audio chunk {i//chunk_size + 1}, size: {len(chunk)} bytes")
                    yield chunk
        
        # Perform streaming recognition
        logger.info(f"Starting streaming recognition with {len(audio_data)} bytes")
        responses = asr_service.streaming_response_generator(
            audio_chunks=audio_chunks_generator(),
            streaming_config=streaming_config
        )
        
        # Collect results
        final_transcript = ""
        best_confidence = 0.0
        segments = []
        partial_results = []
        
        for response in responses:
            if response.results:
                for result in response.results:
                    if result.alternatives:
                        transcript = result.alternatives[0].transcript
                        confidence = result.alternatives[0].confidence
                        
                        if result.is_final:
                            logger.info(f"Final result: '{transcript}' (confidence: {confidence:.3f})")
                            final_transcript += transcript + " "
                            if confidence > best_confidence:
                                best_confidence = confidence
                            
                            # Add as segment
                            alternatives = [
                                TranscriptionAlternative(
                                    transcript=alt.transcript,
                                    confidence=alt.confidence
                                )
                                for alt in result.alternatives
                            ]
                            
                            segment = TranscriptionSegment(
                                alternatives=alternatives,
                                best_transcript=transcript,
                                best_confidence=confidence
                            )
                            segments.append(segment)
                        else:
                            logger.debug(f"Interim result: '{transcript}' (confidence: {confidence:.3f})")
                            partial_results.append((transcript, confidence))
        
        processing_time = (time.time() - start_time) * 1000
        final_transcript = final_transcript.strip()
        
        # If no final results but have partials, use best partial
        if not final_transcript and partial_results:
            best_partial = max(partial_results, key=lambda x: x[1])
            final_transcript = best_partial[0]
            best_confidence = best_partial[1]
            logger.info(f"Using best partial result: '{final_transcript}' (confidence: {best_confidence:.3f})")
        
        logger.info(f"Streaming recognition completed in {processing_time:.2f}ms")
        
        return TranscriptionResponse(
            success=True,
            transcript=final_transcript,
            confidence=best_confidence,
            segments=segments,
            processing_mode="streaming",
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        error_msg = f"Streaming speech recognition failed: {e}"
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return TranscriptionResponse(
            success=False,
            error_message=error_msg,
            processing_mode="streaming"
        )


@mcp.tool()
def recognize_speech(audio_data_base64: str, language_code: Optional[str] = None, mode: Optional[str] = None) -> Dict[str, Any]:
    """
    Perform speech recognition on audio data using NVIDIA Riva ASR.
    
    Args:
        audio_data_base64 (str): Base64 encoded mono WAV audio data (16kHz, 16-bit)
        language_code (Optional[str]): Language code for recognition (default: from config)
        mode (Optional[str]): Recognition mode - 'offline' or 'streaming' (default: from config)
        
    Returns:
        Dict[str, Any]: Recognition results with transcript, confidence, and metadata
    """
    try:
        logger.info("Starting speech recognition")
        
        # Decode base64 audio data
        try:
            audio_data = base64.b64decode(audio_data_base64)
            logger.info(f"Successfully decoded {len(audio_data)} bytes from base64")
        except Exception as e:
            error_msg = f"Failed to decode base64 audio data: {e}"
            logger.error(error_msg)
            return TranscriptionResponse(
                success=False,
                error_message=error_msg
            ).dict()
        
        if len(audio_data) == 0:
            error_msg = "Decoded audio data is empty"
            logger.error(error_msg)
            return TranscriptionResponse(
                success=False,
                error_message=error_msg
            ).dict()
        
        # Use provided parameters or fall back to config
        lang_code = language_code or config.riva_language_code
        asr_mode = (mode or config.riva_asr_mode).lower()
        
        if asr_mode not in ['offline', 'streaming']:
            logger.warning(f"Invalid ASR mode '{asr_mode}', defaulting to 'offline'")
            asr_mode = 'offline'
        
        logger.info(f"Using language: {lang_code}, mode: {asr_mode}")
        
        # Import NVIDIA Riva client
        try:
            import riva.client
            logger.info("Successfully imported NVIDIA Riva client")
        except ImportError as e:
            error_msg = f"Failed to import nvidia-riva-client: {e}"
            logger.error(error_msg)
            return TranscriptionResponse(
                success=False,
                error_message=error_msg
            ).dict()
        
        # Initialize Riva client
        try:
            logger.info(f"Connecting to Riva server at: {config.riva_uri}")
            auth = riva.client.Auth(uri=config.riva_uri)
            asr_service = riva.client.ASRService(auth)
            logger.info("Successfully initialized Riva client")
        except Exception as e:
            error_msg = f"Failed to connect to Riva server at {config.riva_uri}: {e}"
            logger.error(error_msg)
            return TranscriptionResponse(
                success=False,
                error_message=error_msg
            ).dict()
        
        # Write audio to temporary file for Riva processing
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name
        
        try:
            # Configure recognition settings
            logger.info("Configuring recognition settings...")
            riva_config = riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                language_code=lang_code,
                max_alternatives=config.riva_max_alternatives,
                enable_automatic_punctuation=config.riva_enable_punctuation,
                verbatim_transcripts=config.riva_verbatim_transcripts,
            )
            
            # Add audio file specifications
            riva.client.add_audio_file_specs_to_config(riva_config, temp_path)
            logger.info(f"Audio config - Sample rate: {riva_config.sample_rate_hertz}Hz, Channels: {riva_config.audio_channel_count}")
            
            # Read audio data for processing
            with open(temp_path, 'rb') as f:
                audio_bytes = f.read()
            
            # Perform recognition based on mode
            if asr_mode == 'streaming':
                result = transcribe_with_riva_streaming(asr_service, riva_config, audio_bytes)
            else:
                result = transcribe_with_riva_offline(asr_service, riva_config, audio_bytes)
            
            # Add audio duration estimate (rough calculation)
            if riva_config.sample_rate_hertz and riva_config.audio_channel_count:
                sample_width = 2  # 16-bit = 2 bytes
                duration_seconds = len(audio_bytes) / (riva_config.sample_rate_hertz * riva_config.audio_channel_count * sample_width)
                result.audio_duration_ms = duration_seconds * 1000
            
            return result.dict()
            
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
        
    except Exception as e:
        error_msg = f"Unexpected error during speech recognition: {e}"
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return TranscriptionResponse(
            success=False,
            error_message=error_msg
        ).dict()


@mcp.tool()
def get_riva_config() -> Dict[str, Any]:
    """
    Get current NVIDIA Riva configuration.
    
    Returns:
        Dict[str, Any]: Current Riva server configuration
    """
    return {
        "server_name": "Riva Speech Recognition MCP Server",
        "version": "0.1.0",
        "riva_uri": config.riva_uri,
        "language_code": config.riva_language_code,
        "asr_mode": config.riva_asr_mode,
        "max_alternatives": config.riva_max_alternatives,
        "enable_punctuation": config.riva_enable_punctuation,
        "verbatim_transcripts": config.riva_verbatim_transcripts,
        "supported_modes": ["offline", "streaming"],
        "supported_languages": ["en-US", "es-US", "de-DE", "fr-FR", "pt-BR", "zh-CN", "ja-JP", "ko-KR"]
    }


@mcp.tool()
def test_riva_connection() -> Dict[str, Any]:
    """
    Test connection to NVIDIA Riva server.
    
    Returns:
        Dict[str, Any]: Connection test results
    """
    try:
        import riva.client
        
        logger.info(f"Testing connection to Riva server at: {config.riva_uri}")
        
        # Try to initialize connection
        auth = riva.client.Auth(uri=config.riva_uri)
        asr_service = riva.client.ASRService(auth)
        
        # If we get here without exception, connection is successful
        logger.info("Riva connection test successful")
        
        return {
            "success": True,
            "riva_uri": config.riva_uri,
            "status": "connected",
            "message": "Successfully connected to NVIDIA Riva server"
        }
        
    except ImportError as e:
        error_msg = f"NVIDIA Riva client not available: {e}"
        logger.error(error_msg)
        return {
            "success": False,
            "riva_uri": config.riva_uri,
            "status": "client_error",
            "message": error_msg
        }
    except Exception as e:
        error_msg = f"Failed to connect to Riva server: {e}"
        logger.error(error_msg)
        return {
            "success": False,
            "riva_uri": config.riva_uri,
            "status": "connection_error",
            "message": error_msg
        }


def setup_health_endpoint():
    """Set up health check endpoint for the FastAPI app."""
    try:
        app = mcp.streamable_http_app()
        
        @app.get("/health")
        async def health_check():
            """Health check endpoint for Docker and monitoring systems."""
            from datetime import datetime
            return {
                "status": "healthy",
                "service": "Riva Speech Recognition MCP Server",
                "version": "0.1.0",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "tools": ["recognize_speech", "get_riva_config", "test_riva_connection"],
                "riva_uri": config.riva_uri
            }
        
        logger.info("Health check endpoint configured at /health")
        return app
    except Exception as e:
        logger.warning(f"Could not set up health endpoint: {e}")
        return None


def main():
    """Main entry point for the server."""
    parser = argparse.ArgumentParser(description="Riva Speech Recognition MCP Server")
    parser.add_argument(
        "--port", 
        type=int, 
        default=config.port, 
        help=f"Port to run the server on (default: {config.port})"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default=config.host, 
        help=f"Host to bind the server to (default: {config.host})"
    )
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config.host = args.host
    config.port = args.port
    
    print(f"Starting Riva Speech Recognition MCP Server on http://{args.host}:{args.port}")
    print(f"MCP endpoint will be available at: http://{args.host}:{args.port}/mcp")
    print(f"Configured Riva server: {config.riva_uri}")
    print(f"ASR mode: {config.riva_asr_mode}")
    print(f"Language: {config.riva_language_code}")
    
    import uvicorn
    
    # Set up health endpoint and get the app
    app = setup_health_endpoint()
    if app is None:
        app = mcp.streamable_http_app()
    
    # Run with uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=config.log_level.lower()
    )


if __name__ == "__main__":
    main()