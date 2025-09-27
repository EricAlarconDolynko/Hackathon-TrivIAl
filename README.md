# 🩺 Hackathon-TrivIAl - Sistema de Preconsulta Médica con IA

Un sistema inteligente de preconsulta médica que utiliza LangGraph y Cohere para realizar anamnesis automatizada y proporcionar diagnósticos probabilísticos. Este proyecto fue desarrollado para el Hackathon TrivIAl.

##  Cómo Ejecutar el Proyecto

### Prerrequisitos
- Python 3.8 o superior
- Una API key de Cohere 


### Instalación y Ejecución

1. **Clona el repositorio:**
   ```bash
   git clone <tu-repositorio>
   cd Hackathon-TrivIAl
   ```

2. **Instala las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configura las variables de entorno:**
   Crea un archivo `.env` en la raíz del proyecto con tu API key de Cohere:
   ```env
   COHERE_API_KEY=Aae7JZxaPDeyZ3MnuoFWhmguQkPZuhW4zjfRhp5S
   COHERE_MODEL=command-a-03-2025
   HUGGING_API_KEY=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
   ```

4. **Ejecuta la aplicación:**
   ```bash
   streamlit run app.py
   ```

5. **Accede a la aplicación:**
   Se abrirá automáticamente en tu navegador en `http://localhost:8501`

## 📋 ¿Qué Hace el Sistema?

### Funcionalidades Principales

- **Anamnesis Automatizada**: Realiza una entrevista médica estructurada para recopilar información del paciente
- **Diagnóstico Probabilístico**: Utiliza IA para analizar síntomas y sugerir posibles condiciones médicas
- **Interfaz Conversacional**: Interfaz amigable de chat que guía al paciente a través del proceso
- **Validación de Datos**: Asegura que se recopile toda la información necesaria antes de generar diagnósticos

### Flujo del Sistema

1. **Configuración Inicial**: El usuario proporciona edad y nacionalidad
2. **Recopilación de Síntomas**: El sistema pregunta sobre síntomas, severidad, frecuencia y duración
3. **Historia Médica**: Recopila antecedentes personales y familiares
4. **Hábitos de Vida**: Pregunta sobre hábitos relevantes para el diagnóstico
5. **Análisis con IA**: Utiliza Cohere y un clasificador especializado para analizar la información
6. **Diagnóstico Probabilístico**: Genera una lista de posibles condiciones con probabilidades
7. **Preguntas Adicionales**: Realiza preguntas de seguimiento para refinar el diagnóstico
8. **Resultado Final**: Presenta el diagnóstico con recomendaciones

##  Arquitectura del Proyecto

### Estructura de Archivos

```
Hackathon-TrivIAl/
├── app.py                    # Aplicación principal de Streamlit
├── main.py                   # Versión CLI del sistema
├── agent/                    # Módulo del agente inteligente
│   ├── graph.py             # Grafo principal de LangGraph
│   └── providers.py         # Configuración de proveedores de IA
├── artifacts/               # Archivos de despliegue
│   ├── requirements.txt     # Dependencias del proyecto
│   └── shared/             # Módulos compartidos
└── lambda_handler/          # Manejador para AWS Lambda
```

### Tecnologías Utilizadas

- **LangGraph**: Para la orquestación del flujo conversacional
- **Cohere**: Modelo de lenguaje para procesamiento de texto natural
- **Streamlit**: Interfaz web interactiva
- **AWS Lambda**: Para despliegue en la nube
- **Python**: Lenguaje principal del proyecto

##  Configuración Avanzada

### Variables de Entorno Opcionales

```env
COHERE_API_KEY=tu_api_key_aqui          # Requerida
COHERE_MODEL=command-r                  # Modelo de Cohere a usar
COHERE_TEMPERATURE=0.7                  # Temperatura del modelo
COHERE_MAX_TOKENS=1000                  # Máximo de tokens por respuesta
CLASSIFIER_API_URL=url_del_clasificador # URL del clasificador médico
HUGGING_API_KEY=tu_hugging_key          # Para modelos adicionales
```

