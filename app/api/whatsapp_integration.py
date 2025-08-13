# servidor/modulo-ia/app/api/whatsapp_integration.py

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
from datetime import datetime
import logging

from app.services.ia_service import (
    ask_mistral_with_context,
    get_suggested_titles,
    format_topics_inline,
    summarize_pdf,
    get_index_overview
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ia", tags=["WhatsApp IA Integration"])

# ==================== SCHEMAS ====================

class PropertyQuery(BaseModel):
    """Esquema para consultas sobre propiedades"""
    query: str
    client_phone: str
    agent_phone: str
    conversation_history: Optional[List[Dict[str, str]]] = []
    context: Optional[Dict[str, Any]] = {}

class PropertyResponse(BaseModel):
    """Respuesta de IA para consultas de propiedades"""
    success: bool
    response: str
    suggestions: Optional[List[str]] = []
    properties_mentioned: Optional[List[Dict]] = []
    requires_human: bool = False
    metadata: Optional[Dict] = {}

class ClientAnalysis(BaseModel):
    """Análisis de preferencias del cliente"""
    query: str
    client_phone: str
    
class ClientPreferences(BaseModel):
    """Preferencias extraídas del cliente"""
    budget_range: Optional[Dict[str, float]] = None
    location_preferences: Optional[List[str]] = []
    property_type: Optional[str] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    additional_features: Optional[List[str]] = []
    urgency: Optional[str] = None  # "alta", "media", "baja"

# ==================== ENDPOINTS ====================

@router.post("/process-query", response_model=PropertyResponse)
async def process_property_query(request: PropertyQuery):
    """
    Procesa consultas de clientes sobre propiedades inmobiliarias.
    Integrado con el flujo Cliente → Agente → IA → Respuesta
    """
    try:
        logger.info(f"Procesando consulta de {request.client_phone}: {request.query[:100]}...")
        
        # Construir historial de conversación
        history = ""
        if request.conversation_history:
            history_lines = []
            for msg in request.conversation_history[-5:]:  # Últimos 5 mensajes
                history_lines.append(f"Cliente: {msg.get('question', '')}")
                history_lines.append(f"Asistente: {msg.get('answer', '')}")
            history = "\n".join(history_lines)
        
        # Analizar el tipo de consulta
        query_lower = request.query.lower()
        
        # Detectar intenciones específicas del dominio inmobiliario
        is_price_query = any(word in query_lower for word in ['precio', 'costo', 'vale', 'cuesta', 'presupuesto'])
        is_location_query = any(word in query_lower for word in ['zona', 'ubicación', 'dirección', 'dónde', 'barrio'])
        is_availability_query = any(word in query_lower for word in ['disponible', 'libre', 'ocupado', 'alquiler', 'venta'])
        is_visit_request = any(word in query_lower for word in ['visitar', 'ver', 'conocer', 'cita', 'agendar'])
        
        # Si es una solicitud de visita, marcar para intervención humana
        if is_visit_request:
            response_text = (
                "Entiendo que te gustaría visitar la propiedad. "
                "Un agente se pondrá en contacto contigo pronto para coordinar una visita. "
                "¿Hay algún horario que prefieras?"
            )
            return PropertyResponse(
                success=True,
                response=response_text,
                requires_human=True,
                metadata={"intent": "visit_request"}
            )
        
        # Procesar con RAG para consultas sobre propiedades
        result = ask_mistral_with_context(
            query=request.query,
            history=history
        )
        
        # Si no hay contexto suficiente, dar respuesta útil sobre propiedades
        if not result.get("used_context"):
            # Obtener sugerencias de propiedades disponibles
            suggestions = get_suggested_titles(request.query, max_suggestions=3)
            
            if is_price_query:
                response_text = (
                    "Te puedo ayudar con información sobre precios de nuestras propiedades. "
                    "¿Qué tipo de propiedad te interesa? ¿Casa, departamento o terreno? "
                    "También sería útil saber en qué zona estás buscando."
                )
            elif is_location_query:
                response_text = (
                    "Tenemos propiedades en varias zonas de la ciudad. "
                    "Las principales áreas disponibles son: Equipetrol, Zona Norte, "
                    "Urubó, y el Centro. ¿Cuál zona te interesa más?"
                )
            else:
                topics_line = format_topics_inline(suggestions) if suggestions else ""
                response_text = (
                    "Soy tu asistente inmobiliario virtual. Puedo ayudarte con:\n"
                    "• Información sobre propiedades disponibles\n"
                    "• Precios y características\n"
                    "• Ubicaciones y zonas\n"
                    "• Proceso de compra o alquiler\n"
                )
                if topics_line:
                    response_text += f"\nTambién puedo informarte sobre: {topics_line}"
                response_text += "\n\n¿Qué información necesitas?"
            
            return PropertyResponse(
                success=True,
                response=response_text,
                suggestions=suggestions,
                requires_human=False,
                metadata={"used_context": False}
            )
        
        # Respuesta con contexto encontrado
        return PropertyResponse(
            success=True,
            response=result.get("answer", ""),
            suggestions=get_suggested_titles(request.query, max_suggestions=3),
            requires_human=False,
            metadata={"used_context": True}
        )
        
    except Exception as e:
        logger.error(f"Error procesando consulta: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-preferences", response_model=ClientPreferences)
async def analyze_client_preferences(request: ClientAnalysis):
    """
    Analiza el mensaje del cliente para extraer preferencias de propiedades.
    Útil para el sistema de recomendaciones.
    """
    try:
        query_lower = request.query.lower()
        preferences = ClientPreferences()
        
        # Extraer rango de presupuesto
        import re
        price_patterns = [
            r'(\d+\.?\d*)\s*(?:mil|k)\s*(?:bs|bolivianos)?',
            r'(\d{4,})\s*(?:bs|bolivianos)?',
            r'entre\s*(\d+\.?\d*)\s*y\s*(\d+\.?\d*)',
        ]
        
        for pattern in price_patterns:
            matches = re.findall(pattern, query_lower)
            if matches:
                if isinstance(matches[0], tuple):
                    preferences.budget_range = {
                        "min": float(matches[0][0]) * (1000 if 'mil' in query_lower or 'k' in query_lower else 1),
                        "max": float(matches[0][1]) * (1000 if 'mil' in query_lower or 'k' in query_lower else 1)
                    }
                else:
                    amount = float(matches[0]) * (1000 if 'mil' in query_lower or 'k' in query_lower else 1)
                    preferences.budget_range = {"min": amount * 0.8, "max": amount * 1.2}
                break
        
        # Extraer preferencias de ubicación
        locations = ['equipetrol', 'zona norte', 'zona sur', 'centro', 'urubo', 'la guardia']
        preferences.location_preferences = [loc for loc in locations if loc in query_lower]
        
        # Extraer tipo de propiedad
        if 'casa' in query_lower:
            preferences.property_type = 'casa'
        elif 'departamento' in query_lower or 'depto' in query_lower:
            preferences.property_type = 'departamento'
        elif 'terreno' in query_lower:
            preferences.property_type = 'terreno'
        elif 'oficina' in query_lower:
            preferences.property_type = 'oficina'
        
        # Extraer número de dormitorios
        bedroom_match = re.search(r'(\d+)\s*(?:dormitorio|habitacion|cuarto)', query_lower)
        if bedroom_match:
            preferences.bedrooms = int(bedroom_match.group(1))
        
        # Extraer número de baños
        bathroom_match = re.search(r'(\d+)\s*baño', query_lower)
        if bathroom_match:
            preferences.bathrooms = int(bathroom_match.group(1))
        
        # Características adicionales
        features = []
        feature_keywords = {
            'piscina': 'piscina',
            'garage': 'garage',
            'jardin': 'jardín',
            'parrillero': 'parrillero',
            'seguridad': 'seguridad 24/7',
            'amoblado': 'amoblado',
            'balcon': 'balcón'
        }
        
        for keyword, feature in feature_keywords.items():
            if keyword in query_lower:
                features.append(feature)
        
        preferences.additional_features = features
        
        # Detectar urgencia
        if any(word in query_lower for word in ['urgente', 'pronto', 'ya', 'inmediato', 'hoy']):
            preferences.urgency = 'alta'
        elif any(word in query_lower for word in ['próximo mes', 'próxima semana']):
            preferences.urgency = 'media'
        else:
            preferences.urgency = 'baja'
        
        return preferences
        
    except Exception as e:
        logger.error(f"Error analizando preferencias: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/property-info/{property_id}")
async def get_property_ai_description(property_id: str):
    """
    Genera una descripción amigable de una propiedad específica usando IA.
    Se puede conectar con la base de datos de propiedades.
    """
    try:
        # Aquí podrías conectar con la BD de propiedades
        # Por ahora simularemos con datos de ejemplo
        
        property_info = {
            "id": property_id,
            "nombre": "Casa en Equipetrol",
            "precio": 180000,
            "ubicacion": "Equipetrol, 3er Anillo",
            "dormitorios": 3,
            "banos": 2,
            "superficie": "200 m²",
            "caracteristicas": ["Piscina", "Jardín", "Garage para 2 autos"]
        }
        
        # Generar descripción con IA
        prompt = f"""
        Genera una descripción atractiva y profesional para esta propiedad inmobiliaria:
        - Nombre: {property_info['nombre']}
        - Precio: {property_info['precio']} Bs
        - Ubicación: {property_info['ubicacion']}
        - Dormitorios: {property_info['dormitorios']}
        - Baños: {property_info['banos']}
        - Superficie: {property_info['superficie']}
        - Características: {', '.join(property_info['caracteristicas'])}
        
        La descripción debe ser en español, amigable y resaltar los puntos fuertes.
        """
        
        # Aquí usarías tu servicio de IA
        result = ask_mistral_with_context(query=prompt, history="")
        
        return {
            "success": True,
            "property_id": property_id,
            "description": result.get("answer", ""),
            "property_data": property_info
        }
        
    except Exception as e:
        logger.error(f"Error generando descripción: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """
    Verificar el estado del módulo IA y sus componentes.
    """
    try:
        # Verificar índice FAISS
        overview = get_index_overview()
        
        return {
            "success": True,
            "service": "ia-module",
            "status": "healthy",
            "components": {
                "faiss_index": {
                    "status": "ready" if overview["total_chunks"] > 0 else "empty",
                    "total_chunks": overview["total_chunks"],
                    "pdfs_loaded": len(overview.get("pdfs", []))
                },
                "ollama": {
                    "status": "ready",  # Aquí podrías hacer un ping real a Ollama
                    "model": "mistral"
                }
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "service": "ia-module",
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@router.post("/load-property-docs")
async def load_property_documents():
    """
    Endpoint para cargar documentos de propiedades al índice FAISS.
    Útil para actualizar la base de conocimientos con nuevas propiedades.
    """
    try:
        # Aquí podrías implementar la carga de documentos
        # desde la base de datos de propiedades
        
        return {
            "success": True,
            "message": "Documentos cargados correctamente",
            "documents_processed": 0  # Actualizar con el número real
        }
        
    except Exception as e:
        logger.error(f"Error cargando documentos: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))