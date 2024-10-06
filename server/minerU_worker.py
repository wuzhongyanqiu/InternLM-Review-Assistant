import os
import json
import copy

from loguru import logger

from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.pipe.OCRPipe import OCRPipe
from magic_pdf.pipe.TXTPipe import TXTPipe
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
import magic_pdf.model as model_config
import os

model_config.__use_inside_model__ = True

# todo: 设备类型选择 （？）

class MinerU:
    def init_model(self):
        from magic_pdf.model.doc_analyze_by_custom_model import ModelSingleton
        try:
            model_manager = ModelSingleton()
            txt_model = model_manager.get_model(False, False)
            logger.info(f"txt_model init final")
            ocr_model = model_manager.get_model(True, False)
            logger.info(f"ocr_model init final")
            return 0
        except Exception as e:
            logger.exception(e)
            return -1


        model_init = init_model()
        logger.info(f"model_init: {model_init}")

    def parse_dir(self, dir_path: str, output_dir: str = None, parse_method: str = 'auto'):
        datas = []
        """
        解析指定文件夹中的所有 PDF 文件，并将它们转换为 Markdown 和图片文件夹。
        
        :param dir_path: 包含 PDF 文件的文件夹路径
        :param output_dir: 输出结果的目录地址，默认为 None。如果提供，将保存到此目录下。
        :param parse_method: 解析方法，支持 'auto', 'ocr', 'txt' 三种，默认为 'auto'
        """
        # 检查文件夹是否存在
        if not os.path.isdir(dir_path):
            logger.error(f"Directory {dir_path} does not exist")
            return

        # 遍历目录中的文件
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            if file_name.endswith('.pdf'):
                pdf_path = file_path
            
                logger.info(f"Processing file: {pdf_path}")
                content_list = self.pdf_parse_main(pdf_path=pdf_path, parse_method=parse_method, output_dir=output_dir)
                data = self.merge_text_by_level(content_list)
                datas.extend(data)

        return datas

    def pdf_parse_main(
        self,
        pdf_path: str,
        parse_method: str = 'auto',
        output_dir: str = None
    ):
        """
        执行从 pdf 转换到 md 的过程，并生成图片文件夹，仅输出 .md 文件和图片文件夹。

        :param pdf_path: .pdf 文件的路径，可以是相对路径，也可以是绝对路径
        :param parse_method: 解析方法， 共 auto、ocr、txt 三种，默认 auto，如果效果不好，可以尝试 ocr
        :param output_dir: 输出结果的目录地址，会生成一个以 pdf 文件名命名的文件夹并保存所有结果
        """
        try:
            pdf_name = os.path.basename(pdf_path).split(".")[0]
            pdf_path_parent = os.path.dirname(pdf_path)

            if output_dir:
                output_path = os.path.join(output_dir, pdf_name)
            else:
                output_path = os.path.join(pdf_path_parent, pdf_name)

            output_image_path = os.path.join(output_path, 'images')

            # 获取图片的父路径，为的是以相对路径保存到 .md
            image_path_parent = os.path.basename(output_image_path)

            pdf_bytes = open(pdf_path, "rb").read()  # 读取 pdf 文件的二进制数据

            # 执行解析步骤
            image_writer, md_writer = DiskReaderWriter(output_image_path), DiskReaderWriter(output_path)

            # 选择解析方式
            if parse_method == "auto":
                jso_useful_key = {"_pdf_type": "", "model_list": []}
                pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)
            elif parse_method == "txt":
                pipe = TXTPipe(pdf_bytes, [], image_writer)
            elif parse_method == "ocr":
                pipe = OCRPipe(pdf_bytes, [], image_writer)
            else:
                logger.error("unknown parse method, only auto, ocr, txt allowed")
                exit(1)

            # 执行分类
            pipe.pipe_classify()
            pipe.pipe_analyze()
            
            # 执行解析
            pipe.pipe_parse()

            # 生成文本和Markdown内容
            content_list = pipe.pipe_mk_uni_format(image_path_parent, drop_mode="none")

            print(content_list)
            
            ##################################保存成md##################################
            # md_content = pipe.pipe_mk_markdown(image_path_parent, drop_mode="none")

            # 只保存Markdown文件和图片文件夹
            # md_writer.write(
            #     content=md_content,
            #     path=f"{pdf_name}.md"
            # )

            # print(md_content)


        except Exception as e:
            logger.exception(e)

        return content_list

    def merge_text_by_level(self, data):
        merged_texts = []
        current_text = ""
        pre_level = None

        for item in data:
            if isinstance(item, dict) and item.get('type'):
                item_type = item.get('type')
                text_level = item.get('text_level')
                
                if text_level is not None:
                    if pre_level is not None:
                        merged_texts.append(current_text.strip())
                        pre_level = None
                    else:
                        pre_level = text_level
                        if item_type == 'text':
                            current_text = item['text'] + "\n"

                else:
                    if item_type == 'text':
                        current_text += item['text'] + "\n"
                    elif item_type == 'image':
                        current_text += "img_path: " + item['img_path'] + "\n"
                    elif item_type == 'equation':
                        current_text += "equation: " + item['text'] + "\n"

            else:
                continue

        if current_text:
            merged_texts.append(current_text.strip())

        return merged_texts

# 测试
if __name__ == '__main__':
    output_dir = "MinerU_output"
    dir_path = "/root/Mock-Interviewer/lagent/server/storage"
    pdf_path = "/root/Mock-Interviewer/lagent/server/storage/minerU_test.pdf"
    minerU = MinerU()
    minerU.init_model()
    data = minerU.parse_dir(dir_path=dir_path, output_dir=output_dir)
    print(data)