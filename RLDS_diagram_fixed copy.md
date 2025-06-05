# RLDS 图表演示 (修复版本)

## RLDS生态系统架构

```mermaid
graph TB
    subgraph Prod["Data Production"]
        A["EnvLogger<br/>Synthetic Data"] 
        B["RLDS Creator<br/>Human Data"]
        C["Custom Producers"]
    end
    
    subgraph Store["Data Storage"]
        D["RLDS Standard Format"]
        E["TFDS Integration"]
        F["Version Control"]
    end
    
    subgraph Process["Data Processing"]
        G["Transform Library"]
        H["Batch Optimization"]
        I["Performance Optimization"]
    end
    
    subgraph Consume["Data Consumption"]
        J["Episode-level Algorithms"]
        K["Step-level Algorithms"]
        L["Analysis Tools"]
    end
    
    Prod --> Store
    Store --> Process  
    Process --> Consume
    
    Store -.-> M["TFDS Global Catalog"]
    M -.-> N["Community Sharing"]
```

## RLDS数据架构

```mermaid
graph TD
    A["RLDS Dataset"] --> B["Episode Collection"]
    B --> C["Individual Episodes"]
    C --> D["Steps Sequence"]
    
    D --> E[Observation]
    D --> F[Action]
    D --> G[Reward]
    D --> H[Discount]
    D --> I[Metadata]
    
    E --> E1[Images]
    E --> E2["Proprio State"]
    E --> E3["Task Info"]
    
    F --> F1["Joint Commands"]
    F --> F2["End-effector Pose"]
    F --> F3["Gripper Commands"]
    
    subgraph Types["Data Types"]
        J["tf.string<br/>Compressed Images"]
        K["tf.float32<br/>Continuous Values"]
        L["tf.int32<br/>Discrete Values"]
        M["tf.bool<br/>Boolean Values"]
    end
```

## RLDS数据结构层次

```mermaid
flowchart LR
    A[Dataset] --> B[Episodes]
    B --> C[Steps]
    C --> D[Fields]
    
    subgraph Episode["Episode Level"]
        E1[episode_id]
        E2[episode_metadata]
        E3["steps collection"]
    end
    
    subgraph Step["Step Level"]
        S1[observation]
        S2[action]
        S3[reward]
        S4[discount]
        S5[is_first]
        S6[is_last]
        S7[is_terminal]
    end
    
    B -.-> Episode
    C -.-> Step
```

## 配置类结构

```mermaid
classDiagram
    class DatasetConfig {
        +string name
        +string version
        +string description
        +dict features
        +dict splits
        +int total_episodes
        +int total_steps
    }
    
    class FeatureConfig {
        +dict observation_space
        +dict action_space
        +float reward_range
        +string task_type
    }
    
    class SplitConfig {
        +float train_ratio
        +float val_ratio
        +float test_ratio
        +int train_episodes
        +int val_episodes
        +int test_episodes
    }
    
    DatasetConfig --> FeatureConfig : contains
    DatasetConfig --> SplitConfig : contains
```

## 数据流转过程

```mermaid
sequenceDiagram
    participant Env as Environment
    participant Logger as EnvLogger
    participant RLDS as RLDS Creator
    participant TFDS as TensorFlow Datasets
    participant Model as ML Model
    
    Env->>Logger: Generate episodes
    Logger->>RLDS: Raw trajectory data
    RLDS->>TFDS: Standardized format
    TFDS->>Model: Training batches
    Model->>Model: Learning process
```

## 修复说明

主要修复内容：
1. **子图名称标准化**：使用英文标识符，避免中文字符导致的解析错误
2. **节点内容格式化**：使用 `<br/>` 换行，避免特殊字符
3. **引用格式统一**：使用双引号包围含空格或特殊字符的内容
4. **连接语法规范**：确保箭头和连接线符合mermaid语法

这些修复确保图表在支持mermaid的环境中能够正确渲染。 