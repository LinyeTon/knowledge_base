from pathlib import Path
import re

from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import logger

# **函数签名**: `load_markdown_and_image_dir(state: dict) -> tuple[str, Path, Path]`
#             **步骤**
#             1. 读取 `md_content` 和 `md_path`
#             2. 校验 `md_path` 是否为空
#             3. 如果 `md_content` 为空，则按 `md_path` 读取文件正文
#             4. 拼接图片目录 `images`
#             5. 返回正文、Markdown 路径和图片目录路径
def load_markdown_and_image_dir(state) -> tuple[str,Path,Path]:
    # 1. 获取参数 md_content md_path
    md_path = state.get("md_path")
    md_content = state.get("md_content")
    # 2. md_path非空校验
    if not md_path:
        logger.error("md_path为空,无法获取图片地址等,业务无法继续!")
        raise ValueError("md_path为空,无法获取图片地址等,业务无法继续!")
    # 3. md_content进行非空校验 / 空给与默认值
    md_path_obj:Path = Path(md_path)
    if not md_content:
        logger.info(f"md_content没有内容,可能从md数据格式过来的!根据md_path二次读取即可!")
        md_content = md_path_obj.read_text(encoding="utf-8")
        if not md_content:
            logger.error(f"从{md_path}读取md_content内容失败,业务无法继续进行!!")
            raise ValueError(f"从{md_path}读取md_content内容失败,业务无法继续进行!!")
    # 4. images对应Path获取
    images_path_obj = md_path_obj.parent / "images"
    # 5. 返回结果
    return md_content,md_path_obj,images_path_obj


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

#  获取图片的上下文 参数: md_content image_path_obj , context_length:int = 100 响应: list[tuple[str,str,tuple[str,str]]]
#             scan_images
#             [ (图片名 erdaye.png , c:/xxx/erdaye.png, (上文,下文))  ,  , , , , ]
#             思路: 从图片文件夹中获取每张图片! 拿这单张图片去md_content中匹配! 匹配到了! 返回对应位置  start - context_length  end + context_length
#             1. 从imgae_path_obj中获取每一个文件
#             2. 遍历循环 -> 文件判断 -> 是不是图片
#             3. 定义这张图片专属的正则规则
#             4. 使用正则在md_content中进行匹配 search 有 只有一个 或者没有
#             5. 没有 -> md_content没有被引用不用识别上下文!
#             6. 有 -> 获取start | end 截取上下文
#             7. 填装数据
#             8. 返回即可
def scan_images(md_content:str,image_path_obj:Path,context_length:int=100) -> list[tuple[str,str,tuple[str,str]]]:
    images_context = []
    # 1. 从image_path_obj中获取每一个文件
    for image_file_obj in image_path_obj.iterdir():
        image_name = image_file_obj.name
        # 判断是不是图片
        if not image_file_obj.suffix in SUPPORTED_IMAGE_EXTENSIONS:
            # 不是图片
            logger.warning(f"文件:{image_name}不是一张图片,无需处理,跳过本次循环!!")
            continue
        #2. 定义这张图片专属的正则规则
        # ![]( 名字 )
        reg = re.compile(r"\!\[.*?\]\(.*?"+re.escape(image_name)+".*?\)")
        match =  reg.search(md_content)

        #3.match校验,不存在,是图片,但是没有引用
        if not match:
            logger.warning(f"图片:{image_name}没有被md内容引用!无需处理,跳过本次循环!!")
            continue

        #4.match中的定位获取上下文数据
        start,end = match.span()  # match . start() end()
        pre_context = md_content[max(start-context_length,0):start]  # start-context < 0  -> 0
        post_context = md_content[end:min(end+context_length,len(md_content))] # end_context> len(max)  -> len(max)
        images_context.append(
            (
                image_name,
                str(image_file_obj),
                (
                    pre_context,
                    post_context
                )
            )
        )
    logger.info(f"完成了图片的上下文提取: {images_context}")
    return images_context



def enrich_markdown_images(state: ImportGraphState) -> ImportGraphState:
    """
    Markdown 图片增强服务：
    1. 扫描 Markdown 中的图片
    2. 调用多模态模型生成图片说明
    3. 上传图片到 MinIO
    4. 替换 Markdown 图片地址并回写 md_content
    """
    return state