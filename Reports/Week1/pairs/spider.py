import os, sys
import requests
from lxml import html
from tqdm import tqdm


base_url = 'http://www.pemberley.com/etext/PandP/'


"""
@intro: 返回具体文本内容
@param url: 需要爬取的页面
@return <List>: 每段为一个列表元素
"""
def get_content(url):
    response = requests.get(url)
    htm = html.fromstring(response.text.replace('<i>', '').replace('</i>', ''))
    content = [x.replace('\xa0', '').strip() for x in htm.xpath('//div[@class="supptext"]//*/text()')]
    # print(content)
    return content


def save_to_file(content, filename):
    with open(filename, 'w', encoding='utf8') as file:
        file.writelines("\n".join(content))

        
def main():
    current_folder_dir = os.path.dirname(os.path.abspath(__file__))
    for chapter in tqdm(range(1, 62, 1), ncols=70):
        content = get_content(base_url + 'chapter{}.htm'.format(chapter))
        file_path = os.path.join(current_folder_dir, "data/chapter{}.txt".format(chapter))
        save_to_file(content, file_path)

if __name__ == '__main__':
    main()
