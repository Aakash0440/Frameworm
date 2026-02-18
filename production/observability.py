"""
Observability with OpenTelemetry tracing.

Example:
    >>> from frameworm.production import setup_tracing
    >>> 
    >>> tracer = setup_tracing('frameworm-api')
    >>> 
    >>> with tracer.start_as_current_span('inference'):
    ...     result = model(input)
"""

from typing import Optional


def setup_tracing(
    service_name: str,
    endpoint: Optional[str] = None
):
    """
    Setup OpenTelemetry tracing.
    
    Args:
        service_name: Name of your service
        endpoint: OTLP endpoint (e.g., 'http://localhost:4317')
        
    Returns:
        Tracer instance
        
    Example:
        >>> tracer = setup_tracing('my-service')
        >>> with tracer.start_as_current_span('train'):
        ...     train_model()
    """
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        print("⚠️  opentelemetry not installed")
        print("   Install: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp")
        return None
    
    # Create resource
    resource = Resource.create({
        "service.name": service_name
    })
    
    # Setup provider
    provider = TracerProvider(resource=resource)
    
    # Add exporter if endpoint provided
    if endpoint:
        exporter = OTLPSpanExporter(endpoint=endpoint)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
    
    trace.set_tracer_provider(provider)
    
    tracer = trace.get_tracer(service_name)
    print(f"✓ OpenTelemetry tracing enabled for {service_name}")
    
    return tracer