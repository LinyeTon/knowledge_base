from dataclasses import dataclass

from minio import Minio

from app.shared.clients.minio_utils import get_minio_client
from app.infra.config.providers import infra_config

# 封装minio的gateway! minio对外提供 属性 和 方法的`网关`
# 对外的属性: bucket_name  image_dir
# 对外的函数: client()  build_image_url()
# @dataclass
class MinioGateway:

    ## bucket_name : str = infra_config.minio.bucket_name

    @property
    def bucket_name(self):
        return infra_config.minio.bucket_name

    @property
    def image_dir(self):
        return infra_config.minio.minio_img_dir


    def client(self):
        # minio_utils
        return get_minio_client()

    def build_image_url(self, stem:str, object_name:str):
        # 桶
           # 文件名
                # 对象名
        #  协议 :// 端点:9000  / 桶 / minio_img_dir /  文件名 / 对象名
        protocol = "https" if infra_config.minio.minio_secure else "http"

        return (
            f"{protocol}://{infra_config.minio.endpoint}/{infra_config.minio.bucket_name}"
            f"{infra_config.minio.minio_img_dir}/{stem}/{object_name}"
        )

minio_gateway = MinioGateway()
