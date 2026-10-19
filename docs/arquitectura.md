# Arquitectura AWS (Tarea 5)

![Arquitectura](arquitectura_aws.png)

Fuente editable: [`arquitectura_aws.drawio`](arquitectura_aws.drawio) (draw.io / diagrams.net, íconos oficiales AWS). También en [`arquitectura_aws.svg`](arquitectura_aws.svg).

## Diagrama (Mermaid)

```mermaid
flowchart LR
    subgraph Fuentes
        ERP[(ERP / OMS<br/>pedidos, capacidad)]
        FB[Pronóstico base<br/>sistema actual]
        APP[App B2B / Web<br/>clientes y agentes]
    end

    subgraph Ingesta_y_Datos["Persistencia · Data Lake"]
        S3R[(S3 raw)]
        GLUE[Glue Jobs + Data Catalog<br/>validación de calidad]
        S3C[(S3 curated<br/>Parquet particionado)]
        ATH[Athena<br/>consultas ad-hoc]
    end

    subgraph Batch["Procesamiento batch / re-entrenamiento"]
        EB[EventBridge Scheduler<br/>diario / semanal]
        SF[Step Functions<br/>orquestación]
        GATE[Lambda: Gate 45d<br/>por CD]
        SMP[SageMaker Pipelines<br/>features → train → backtest]
        REG[SageMaker Model Registry<br/>aprobación por WAPE]
        BT[SageMaker Batch Transform<br/>pronóstico corregido D+1..D+4]
    end

    subgraph Serving["Servicio de recomendación / inferencia"]
        APIGW[API Gateway<br/>auth Cognito/IAM]
        LREC[Lambda / ECS Fargate<br/>motor de fecha recomendada]
        SME[SageMaker Endpoint<br/>serverless]
        DDB[(DynamoDB<br/>pronóstico corregido,<br/>capacidad, estado gate)]
    end

    subgraph Obs["Monitoreo y estados"]
        CW[CloudWatch<br/>logs, alarmas, % fallback]
        MM[SageMaker Model Monitor<br/>drift]
        RDS[(Aurora / RDS<br/>métricas WAPE/Bias,<br/>historial de decisiones)]
        QS[QuickSight<br/>dashboard KPIs]
    end

    ERP --> S3R
    FB --> S3R
    S3R --> GLUE --> S3C --> ATH
    EB --> SF --> GATE --> SMP --> REG --> BT
    S3C --> SMP
    BT --> DDB
    GATE -->|estado por CD| DDB
    APP --> APIGW --> LREC
    LREC --> DDB
    LREC -.on-demand.-> SME
    LREC -->|eventos de pedido| S3R
    SMP --> RDS
    LREC --> CW
    SME --> MM
    RDS --> QS
    CW -->|alarma| SF
```

| Componente | Rol | Interacciones / razonamiento |
|---|---|---|
| **S3 raw / curated + Glue + Athena** | *Persistencia*: data lake con histórico de pedidos, pronósticos, capacidad y plan | Glue valida esquema y calidad (nulos, huecos, negativos) antes de que lleguen al modelo; Parquet particionado por fecha/CD |
| **EventBridge + Step Functions** | *Batch*: orquesta el pipeline diario (inferencia) y semanal (re-entrenamiento) | Reintentos y ramas de error explícitas; si algo falla, se mantiene el último pronóstico válido |
| **Lambda Gate 45d** | Evalúa historia continua por CD y publica el modo (MODEL/FALLBACK) | Su salida decide qué CDs entran al entrenamiento y qué fuente se sirve; queda trazada en DynamoDB y RDS |
| **SageMaker Pipelines + Model Registry** | Feature engineering, entrenamiento, backtest rolling-origin y registro | Solo se promueve un modelo si su WAPE de backtest ≤ el campeón y ≤ fallback (*quality gate* automático) |
| **SageMaker Batch Transform** | Genera el pronóstico corregido D+1..D+4 de todos los CDs cada día | Batch porque el pronóstico es diario y por CD: más barato y simple que inferencia online |
| **DynamoDB** | Estado *caliente*: pronóstico corregido, capacidad por slot y estado del gate | Lectura de ms para el servicio de recomendación |
| **API Gateway + Lambda/Fargate** | *Servicio de recomendación*: al crear un pedido, devuelve las fechas sugeridas con utilización proyectada e incentivo | Lee DynamoDB (no llama al modelo en la ruta crítica → baja latencia y sin dependencia del endpoint). El endpoint serverless queda para re-cálculos on-demand |
| **Aurora/RDS + CloudWatch + Model Monitor + QuickSight** | *Persistencia de métricas y estados*: WAPE/Bias por CD/horizonte, % fallback, decisiones del gate, drift | Alarmas (p.ej., WAPE semanal > umbral o % fallback ↑) disparan re-entrenamiento o rollback |

**Resiliencia end-to-end:** el servicio nunca depende de que el modelo esté disponible: si no hay pronóstico corregido vigente se sirve el fallback (sesgo móvil o base) desde DynamoDB.
