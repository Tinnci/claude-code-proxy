#!/usr/bin/env python3
"""
Comprehensive test suite for Anthropic-Proxy.

This script tests both streaming and non-streaming requests for all major
supported providers (OpenAI, Gemini, Vertex, XAI, OpenRouter) and the
default Claude model mapping.

Usage:
  python tests.py                    # Run all tests against proxy and (if configured) Anthropic
  python tests.py --proxy-only       # Skip calling Anthropic, only test proxy behavior
  python tests.py --no-streaming     # Skip all streaming tests
  python tests.py --provider gemini  # Run only tests for a specific provider (e.g., openai, gemini, xai)
"""

import os
import json
import time
import httpx
import argparse
import asyncio
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
from dotenv import load_dotenv
import importlib.util as _iu

# Load environment variables
load_dotenv()

# -------- helper: safe module detection --------
def has_module(name: str) -> bool:
    """Return True if module (top-level) can be found without importing."""
    try:
        return _iu.find_spec(name) is not None
    except ModuleNotFoundError:
        return False

# Configuration
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
PROXY_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "") # Proxy can reuse key or be empty

# Mark if official Anthropic can be called
ANTHROPIC_ENABLED = bool(ANTHROPIC_API_KEY)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
PROXY_API_URL = "http://localhost:8082/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

# --- Models for provider-specific tests ---
# We use explicit model IDs to bypass the PREFERRED_PROVIDER mapping and test each backend directly.
MODELS_TO_TEST = {
    "claude_mapping": "claude-3-haiku-20240307", # Tests the default mapping logic based on PREFERRED_PROVIDER
    "openai": "openai/gpt-4o-mini",
    "gemini": "gemini/gemini-1.5-flash-latest",
    "vertex_ai": "vertex_ai/gemini-1.5-flash-preview-0514",
    "xai": "xai/grok-3-mini-beta",
    "openrouter": "openrouter/mistralai/mistral-7b-instruct:free",
}

# Provider keys for checking configuration status
PROVIDER_KEYS = {
    "openai": bool(os.getenv("OPENAI_API_KEY")),
    "gemini": bool(os.getenv("GEMINI_API_KEY")),
    "vertex_ai": bool(os.getenv("VERTEX_PROJECT_ID") and has_module("google.auth")),
    "xai": bool(os.getenv("XAI_API_KEY")),
    "openrouter": bool(os.getenv("OPENROUTER_API_KEY")),
    "claude_mapping": True, # Always runnable
}


# Headers
anthropic_headers = {
    "x-api-key": ANTHROPIC_API_KEY,
    "anthropic-version": ANTHROPIC_VERSION,
    "content-type": "application/json",
}

proxy_headers = {
    "x-api-key": PROXY_API_KEY,
    "anthropic-version": ANTHROPIC_VERSION,
    "content-type": "application/json",
}

# Tool definitions
calculator_tool = {
    "name": "calculator",
    "description": "Evaluate mathematical expressions",
    "input_schema": {
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "The mathematical expression to evaluate"}},
        "required": ["expression"]
    }
}

# Test scenarios
def generate_test_scenarios():
    """Generate test scenarios for each provider model."""
    scenarios = {}

    # 1. Simple text generation test for each provider
    for provider, model_id in MODELS_TO_TEST.items():
        scenarios[f"simple_{provider}"] = {
            "model": model_id,
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Hello, please say three words."}]
        }

    # 2. A more complex tool use test (using a reliable model)
    scenarios["complex_tool_use"] = {
        "model": "claude-3-sonnet-20240229",
        "max_tokens": 500,
        "messages": [{"role": "user", "content": "What is 24 * 7?"}],
        "tools": [calculator_tool],
        "tool_choice": {"type": "auto"}
    }
    
    # 3. Streaming test for each provider
    for provider, model_id in MODELS_TO_TEST.items():
         scenarios[f"stream_{provider}"] = {
            "model": model_id,
            "max_tokens": 10,
            "stream": True,
            "messages": [{"role": "user", "content": "Hello, count to 3."}]
        }
        
    return scenarios

TEST_SCENARIOS = generate_test_scenarios()


# Required event types for a valid Anthropic-like stream
REQUIRED_EVENT_TYPES = {"message_start", "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"}

# ================= NON-STREAMING TESTS =================

def get_response(url, headers, data):
    """Send a request and get the response."""
    try:
        response = httpx.post(url, headers=headers, json=data, timeout=60)
        return response
    except Exception as e:
        print(f"Error sending request to {url}: {e}")
        return None

def compare_responses(anthropic_response, proxy_response):
    """Compare the two responses to see if they're similar enough."""
    proxy_json = proxy_response.json()
    
    # In proxy-only mode, we can't compare to Anthropic, just check structure
    if not anthropic_response:
        assert proxy_json.get("role") == "assistant"
        assert proxy_json.get("content") and isinstance(proxy_json["content"], list)
        return True

    anthropic_json = anthropic_response.json()
    
    print("\n--- Anthropic Response Structure ---")
    print(json.dumps({k: v for k, v in anthropic_json.items() if k != "content"}, indent=2))
    
    print("\n--- Proxy Response Structure ---")
    print(json.dumps({k: v for k, v in proxy_json.items() if k != "content"}, indent=2))
    
    assert proxy_json.get("role") == "assistant", "Proxy role is not 'assistant'"
    assert proxy_json.get("type") == "message", "Proxy type is not 'message'"
    assert "content" in proxy_json and isinstance(proxy_json["content"], list), "Proxy content is invalid"
    assert len(proxy_json["content"]) > 0, "Proxy content is empty"
    
    return True

def test_request(test_name, request_data):
    """Run a test with the given request data."""
    print(f"\n{'='*20} RUNNING TEST: {test_name} {'='*20}")
    
    # Log the request data
    print(f"Requesting model: {request_data.get('model')}")
    
    proxy_data = request_data.copy()
    
    try:
        # Determine expected status code in proxy-only mode
        if not ANTHROPIC_ENABLED:
            provider = test_name.split('_')[-1]
            if provider == "vertex_ai" and not PROVIDER_KEYS["vertex_ai"]:
                expected_status = 501 # Not configured
            elif not PROVIDER_KEYS.get(provider, True):
                 # If provider key is missing, litellm will fail inside the proxy
                expected_status = 500
            else:
                expected_status = 200 # Should succeed
        
        # Send request to Proxy
        print("\nSending to Proxy...")
        proxy_response = get_response(PROXY_API_URL, proxy_headers, proxy_data)
        
        # Validation for proxy-only mode
        if not ANTHROPIC_ENABLED:
            if proxy_response and proxy_response.status_code == expected_status:
                print(f"✅ Proxy returned expected status code {expected_status} for provider '{provider}'.")
                return True
            else:
                status = proxy_response.status_code if proxy_response else 'no response'
                print(f"❌ Proxy returned {status} for '{provider}', expected {expected_status}.")
                return False

        # Validation when comparing to Anthropic
        print("\nSending to Anthropic API...")
        anthropic_response = get_response(ANTHROPIC_API_URL, anthropic_headers, request_data)

        print(f"\nAnthropic status code: {anthropic_response.status_code if anthropic_response else 'N/A'}")
        print(f"Proxy status code: {proxy_response.status_code if proxy_response else 'N/A'}")

        if not (anthropic_response and proxy_response and proxy_response.status_code == 200):
            print("❌ Test failed: One or both requests failed or proxy did not return 200.")
            return False
            
        result = compare_responses(anthropic_response, proxy_response)
        
        if result:
            print(f"\n✅ Test {test_name} passed!")
            return True
        else:
            print(f"\n❌ Test {test_name} failed!")
            return False
    
    except Exception as e:
        print(f"\n❌ Error in test {test_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# ================= STREAMING TESTS =================

class StreamStats:
    """Track statistics about a streaming response."""
    
    def __init__(self):
        self.event_types = set()
        self.text_content = ""
        self.has_tool_use = False
        self.has_error = False
        self.error_message = ""
        
    def add_event(self, event_data):
        """Track information about each received event."""
        if "type" in event_data:
            event_type = event_data["type"]
            self.event_types.add(event_type)
            
            if event_type == "content_block_start" and event_data.get("content_block", {}).get("type") == "tool_use":
                self.has_tool_use = True
            elif event_type == "content_block_delta" and event_data.get("delta", {}).get("type") == "text_delta":
                self.text_content += event_data["delta"].get("text", "")

    def summarize(self):
        """Print a summary of the stream statistics."""
        print(f"Unique event types: {sorted(list(self.event_types))}")
        print(f"Has tool use: {self.has_tool_use}")
        if self.text_content: print(f"Text preview: '{self.text_content[:70]}...'")
        else: print("No text content extracted")
        if self.has_error: print(f"Error: {self.error_message}")

async def stream_response(url, headers, data, stream_name):
    """Send a streaming request and process the response."""
    print(f"\nStarting {stream_name} stream...")
    stats = StreamStats()
    
    try:
        async with httpx.AsyncClient() as client:
            request_data = data.copy()
            request_data["stream"] = True
            
            async with client.stream("POST", url, json=request_data, headers=headers, timeout=60) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    stats.has_error = True
                    stats.error_message = f"HTTP {response.status_code}: {error_text.decode('utf-8', 'ignore')}"
                    print(f"Error: {stats.error_message}")
                    return stats
                
                print(f"{stream_name} connected, receiving events...")
                buffer = ""
                async for chunk in response.aiter_text():
                    buffer += chunk
                    # Process server-sent events
                    while "\n\n" in buffer:
                        event_text, buffer = buffer.split("\n\n", 1)
                        if "data: " in event_text:
                            data_part = event_text.split("data: ", 1)[1]
                            if data_part.strip() == "[DONE]": continue
                            try:
                                event_data = json.loads(data_part)
                                stats.add_event(event_data)
                            except json.JSONDecodeError:
                                print(f"Warning: Could not decode JSON from stream chunk: {data_part}")
                
            print(f"{stream_name} stream completed.")
    except Exception as e:
        stats.has_error = True
        stats.error_message = str(e)
        print(f"Error in {stream_name} stream: {e}")
    
    return stats

def compare_stream_stats(anthropic_stats, proxy_stats):
    """Compare the statistics from the two streams to see if they're similar enough."""
    print("\n--- Stream Comparison ---")
    
    if proxy_stats.has_error:
        print(f"❌ Proxy stream reported an error: {proxy_stats.error_message}")
        return False
        
    proxy_has_content = proxy_stats.has_tool_use or len(proxy_stats.text_content) > 0
    
    # In proxy-only mode, check for minimal validity
    if not anthropic_stats:
        if not proxy_has_content:
            print("❌ Proxy-only test failed: stream had no text or tool content.")
            return False
        if not ("content_block_delta" in proxy_stats.event_types or "message_delta" in proxy_stats.event_types):
             print("❌ Proxy-only test failed: stream did not contain any content delta events.")
             return False
        print("✅ Proxy-only stream test seems valid (has content and delta events).")
        return True

    # When comparing to Anthropic
    anthropic_has_content = anthropic_stats.has_tool_use or len(anthropic_stats.text_content) > 0
    if anthropic_has_content and not proxy_has_content:
        print("❌ FAILED: Anthropic stream had content, but proxy stream had none.")
        return False

    print("✅ Stream content check passed.")
    return True

async def test_streaming(test_name, request_data):
    """Run a streaming test with the given request data."""
    print(f"\n{'='*20} RUNNING STREAMING TEST: {test_name} {'='*20}")
    print(f"Requesting model: {request_data.get('model')}")
    
    proxy_data = request_data.copy()
    
    try:
        anthropic_stats = None
        if ANTHROPIC_ENABLED:
            anthropic_stats = await stream_response(ANTHROPIC_API_URL, anthropic_headers, request_data, "Anthropic")
        
        proxy_stats = await stream_response(PROXY_API_URL, proxy_headers, proxy_data, "Proxy")
        
        print("\n--- Anthropic Stream Statistics ---")
        if anthropic_stats: anthropic_stats.summarize()
        else: print("N/A (not run)")
        
        print("\n--- Proxy Stream Statistics ---")
        proxy_stats.summarize()
        
        result = compare_stream_stats(anthropic_stats, proxy_stats)
        
        # Stricter check for proxy-only: if a provider is configured, it should not error.
        if not ANTHROPIC_ENABLED:
            provider = test_name.split('_')[-1]
            if PROVIDER_KEYS.get(provider) and proxy_stats.has_error:
                print(f"❌ Configured provider '{provider}' should not have errored in stream.")
                result = False

        if result:
            print(f"\n✅ Test {test_name} passed!")
            return True
        else:
            print(f"\n❌ Test {test_name} failed!")
            return False
    
    except Exception as e:
        print(f"\n❌ Error in test {test_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# ================= MAIN =================

async def run_tests(args):
    """Run all tests based on command-line arguments."""
    results = {}
    
    all_tests = list(TEST_SCENARIOS.items())
    
    for test_name, test_data in all_tests:
        # Filter by provider if specified
        if args.provider and args.provider not in test_name:
            continue
            
        is_streaming_test = test_data.get("stream", False)
        
        # Handle streaming/non-streaming flags
        if is_streaming_test and args.no_streaming:
            continue
        if not is_streaming_test and args.streaming_only:
            continue
            
        if is_streaming_test:
            result = await test_streaming(test_name, test_data)
        else:
            result = test_request(test_name, test_data)
        results[test_name] = result
    
    # Print summary
    print("\n\n=========== TEST SUMMARY ===========\n")
    total = len(results)
    if total == 0:
        print("No tests were run. Check your filter arguments.")
        return True
        
    passed = sum(1 for v in results.values() if v)
    
    for test, result in results.items():
        print(f"{test}: {'✅ PASS' if result else '❌ FAIL'}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠️ {total - passed} tests failed")
        return False

async def main():
    parser = argparse.ArgumentParser(description="Test the Anthropic-Proxy server")
    parser.add_argument("--no-streaming", action="store_true", help="Skip streaming tests")
    parser.add_argument("--streaming-only", action="store_true", help="Only run streaming tests")
    parser.add_argument("--proxy-only", action="store_true", help="Skip calling Anthropic, only test proxy")
    parser.add_argument("--provider", type=str, help="Run only tests for a specific provider (e.g., openai, gemini)")
    args = parser.parse_args()

    global ANTHROPIC_ENABLED
    if args.proxy_only:
        ANTHROPIC_ENABLED = False

    if not ANTHROPIC_ENABLED:
        print("🔄 Running in proxy-only mode.")
    else:
        print("🔄 Running in comparison mode (Proxy vs. Anthropic).")

    # If comparison mode is on but key is missing
    if ANTHROPIC_ENABLED and not ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY not set. Use --proxy-only to run without it.")
        sys.exit(1)

    # Run tests
    success = await run_tests(args)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())
