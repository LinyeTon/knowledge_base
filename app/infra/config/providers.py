from app.shared.config.embedding_config import embedding_config, EmbeddingConfig
from app.shared.config.lm_config import lm_config, LLMConfig
from app.shared.config.bailian_mcp_config import mcp_config, McpConfig
from app.shared.config.milvus_config import milvus_config, MilvusConfig
from app.shared.config.mineru_config import mineru_config, MinerUConfig
from app.shared.config.minio_config import minio_config, MinIOConfig
from app.shared.config.reranker_config import reranker_config, RerankerConfig
from app.shared.config.settings_config import AppSettings, settings

from dataclasses import dataclass , field

@dataclass
class InfraConfig:
    #属性名: 类型 =  可变类型 对象 集合 字典等等 不能直接赋值
    # app -> 函数(default_factory) ->  函数的返回值 两个地址
    app: AppSettings = field(default_factory=lambda :  settings)
    llm: LLMConfig = field(default_factory=lambda: lm_config)
    embedding: EmbeddingConfig = field(default_factory=lambda: embedding_config)
    reranker: RerankerConfig = field(default_factory=lambda: reranker_config)
    mcp: McpConfig = field(default_factory=lambda: mcp_config)
    milvus: MilvusConfig = field(default_factory=lambda: milvus_config)
    mineru: MinerUConfig = field(default_factory=lambda: mineru_config)
    minio: MinIOConfig = field(default_factory=lambda: minio_config)

infra_config = InfraConfig()
print(infra_config.llm.llm_model)