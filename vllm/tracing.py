# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Mapping
from typing import Optional

from vllm.logger import init_logger
from vllm.utils import run_once

TRACE_HEADERS = ["traceparent", "tracestate", "SOFA-TraceId", "SOFA-RpcId", "X-Request-ID", "X-AIGW-APP-KeyId"]

APP_NAME = 'vllm'

NORMAL_TRACE_LEVEL = 1
ENHANCED_TRACE_LEVEL = 2

logger = init_logger(__name__)

_is_otel_imported = False
otel_import_error_traceback: Optional[str] = None
try:
    from opentelemetry.context.context import Context
    from opentelemetry.sdk.environment_variables import (
        OTEL_EXPORTER_OTLP_TRACES_PROTOCOL)
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider, Status, StatusCode
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import SpanKind, Tracer, set_tracer_provider
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator)
    _is_otel_imported = True
except ImportError:
    # Capture and format traceback to provide detailed context for the import
    # error. Only the string representation of the error is retained to avoid
    # memory leaks.
    # See https://github.com/vllm-project/vllm/pull/7266#discussion_r1707395458
    import traceback
    otel_import_error_traceback = traceback.format_exc()

    class Context:  # type: ignore
        pass

    class BaseSpanAttributes:  # type: ignore
        pass

    class SpanKind:  # type: ignore
        pass

    class Tracer:  # type: ignore
        pass

    class Status:  # type: ignore
        pass

    class StatusCode:  # type: ignore
        pass


def is_otel_available() -> bool:
    return _is_otel_imported


def init_tracer(instrumenting_module_name: str,
                otlp_traces_endpoint: str) -> Optional[Tracer]:
    if not is_otel_available():
        raise ValueError(
            "OpenTelemetry is not available. Unable to initialize "
            "a tracer. Ensure OpenTelemetry packages are installed. "
            f"Original error:\n{otel_import_error_traceback}")
    resource = Resource.create({"service.name": "vllm"})
    trace_provider = TracerProvider(resource=resource)
    set_tracer_provider(trace_provider)

    span_exporter = get_span_exporter(otlp_traces_endpoint)
    trace_provider.add_span_processor(BatchSpanProcessor(span_exporter))

    tracer = trace_provider.get_tracer(instrumenting_module_name)

    return tracer


def get_span_exporter(endpoint):
    protocol = os.environ.get(OTEL_EXPORTER_OTLP_TRACES_PROTOCOL, "grpc")
    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter)
    elif protocol == "http/protobuf":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter)  # type: ignore
    else:
        raise ValueError(
            f"Unsupported OTLP protocol '{protocol}' is configured")

    return OTLPSpanExporter(endpoint=endpoint)


def extract_trace_context(
        headers: Optional[Mapping[str, str]]) -> Optional[Context]:
    if is_otel_available():
        headers = headers or {}
        return TraceContextTextMapPropagator().extract(headers)
    else:
        return None


def extract_trace_headers(headers: Mapping[str, str]) -> Mapping[str, str]:

    return {h: headers[h] for h in TRACE_HEADERS if h in headers}


class SpanAttributes:
    # Attribute names copied from here to avoid version conflicts:
    # https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md
    GEN_AI_USAGE_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"
    GEN_AI_USAGE_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
    GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
    GEN_AI_REQUEST_TOP_K = "gen_ai.request.top_k"
    GEN_AI_REQUEST_MIN_P = "gen_ai.request.min_p"
    GEN_AI_REQUEST_REPETITION_PENALTY = "gen_ai.request.repetition_penalty"
    GEN_AI_REQUEST_FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
    GEN_AI_REQUEST_PRESENCE_PENALTY = "gen_ai.request.presence_penalty"
    # Attribute names added until they are added to the semantic conventions:
    GEN_AI_REQUEST_ID = "gen_ai.request.id"
    GEN_AI_REQUEST_N = "gen_ai.request.n"
    GEN_AI_RESPONSE_FINISH_REASON = "gen_ai.response.finish_reasons"
    GEN_AI_USAGE_NUM_SEQUENCES = "gen_ai.usage.num_sequences"
    GEN_AI_LATENCY_TIME_IN_QUEUE = "gen_ai.latency.time_in_queue"
    GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN = "gen_ai.latency.time_to_first_token"
    GEN_AI_LATENCY_E2E = "gen_ai.latency.e2e"
    GEN_AI_LATENCY_TIME_IN_SCHEDULER = "gen_ai.latency.time_in_scheduler"
    # Time taken in the forward pass for this across all workers
    GEN_AI_LATENCY_TIME_IN_MODEL_FORWARD = (
        "gen_ai.latency.time_in_model_forward")
    # Time taken in the model execute function. This will include model
    # forward, block/sync across workers, cpu-gpu sync time and sampling time.
    GEN_AI_LATENCY_TIME_IN_MODEL_EXECUTE = (
        "gen_ai.latency.time_in_model_execute")
    GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL = \
        "gen_ai.latency.time_in_model_prefill"
    GEN_AI_LATENCY_TIME_IN_MODEL_DECODE = "gen_ai.latency.time_in_model_decode"
    GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE = \
        "gen_ai.latency.time_in_model_inference"
    GEN_AI_REQUEST_TRACE_LEVEL = "gen_ai.request.trace_level"
    
    # trace_level_2
    GEN_AI_LATENCY_PER_TOKEN_GENERATION_TIME = "gen_ai.latency.per_token_generation_time"
    GEN_AI_LATENCY_PER_TOKEN_SCHEDULED_TIME = "gen_ai.latency.per_token_scheduled_time"
    GEN_AI_ITERATION_PER_TOKEN_BATCH_SIZE = "gen_ai.iteration.per_token_batch_size"
    GEN_AI_ITERATION_PER_TOKEN_TOTAL_TOKENS = "gen_ai.iteration.per_token_total_tokens"
    GEN_AI_RESPONSE_PER_TOKEN_CANDIDATE_DECODED_TOKENS = "gen_ai.response.per_token_candidate_decoded_tokens"
    GEN_AI_RESPONSE_PER_TOKEN_CANDIDATE_TOKEN_IDS = "gen_ai.response.per_token_candidate_token_ids"
    GEN_AI_RESPONSE_PER_TOKEN_CANDIDATE_TOKENS_LOGPROBS = "gen_ai.response.per_token_candidate_tokens_logprobs"

    SOFA_TRACE_ID = "Parent-TraceId"
    SOFA_RPC_ID = "Parent-RpcId"
    REQUEST_ID = "alipay.aicloud.request_id"
    API_KEY_ID = "alipay.aicloud.api_key_id"
    POD_IP = "alipay.base.ip"
    POD_NAME = "alipay.base.pod_name"
    HOSTNAME = "alipay.base.host"
    IDC = "alipay.base.idc"
    MODEL_SERVICE_ID = "alipay.aicloud.model_service_id"
    MODEL_INSTANCE_ID = "alipay.aicloud.model_instance_id"
    MODEL_INSTANCE_NAME = "alipay.aicloud.model_instance_name"
    APP_NAME = "alipay.aicloud.app_name"
    ALIPAY_LATENCY_TIME_IN_API_SERVER = "alipay.aicloud.time_in_api_server"
    ALIPAY_LATENCY_TIME_IN_INPUT_PROCESSING = "alipay.aicloud.time_in_input_processing"
    ALIPAY_LATENCY_TIME_IN_OUTPUT_QUEUE = "alipay.aicloud.time_in_output_queue"
    ALIPAY_LATENCY_TIME_IN_OUTPUT_PROCESSING = "alipay.aicloud.time_in_output_processing"
    ALIPAY_REQUEST_PARAMS = "alipay.aicloud.request_params"


def contains_trace_headers(headers: Mapping[str, str]) -> bool:
    return any(h in headers for h in TRACE_HEADERS)


@run_once
def log_tracing_disabled_warning() -> None:
    logger.warning(
        "Received a request with trace context but tracing is disabled")
