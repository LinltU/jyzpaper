"""Create an A4 concise progress PDF, with embedded CJK fonts and verified image geometry.

Run with system python3 (ReportLab + Pillow). Figure preparation uses jyz_paper.
"""
from pathlib import Path
import hashlib
import html
import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, PageBreak, KeepTogether, Table, TableStyle

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / 'v4_results_20261009/06_reports'
ASSETS = REPORTS / 'V4_PROGRESS_ASSETS'
MD = ROOT / '绝缘子检测算法进展（v4版本）.md'
PDF = MD.with_suffix('.pdf')
FONT = 'ProgressCJK'
PAGE_W, PAGE_H = A4
WIDTH = PAGE_W - 30 * mm


def inline(text):
    escaped = html.escape(text)
    return re.sub(r'[\x20-\x7e\u00b1\u00b7]+', lambda m: f'<font name="DejaVu">{m.group()}</font>', escaped)


def main():
    ASSETS.mkdir(exist_ok=True)
    subprocess.run(['/home/test/miniconda3/envs/jyz_paper/bin/python', '-W', 'error',
                    str(Path(__file__).with_name('prepare_v4_progress_figures.py')),
                    '--output', str(ASSETS)], check=True)
    pdfmetrics.registerFont(TTFont(FONT, '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'))
    pdfmetrics.registerFont(TTFont('DejaVu', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    styles = {
        'body': ParagraphStyle('body', fontName=FONT, fontSize=10, leading=16, wordWrap='CJK', spaceAfter=7),
        'title': ParagraphStyle('title', fontName=FONT, fontSize=19, leading=26, textColor=colors.HexColor('#172554'), spaceAfter=8),
        'subtitle': ParagraphStyle('subtitle', fontName=FONT, fontSize=12, leading=18, textColor=colors.HexColor('#475569'), spaceAfter=12),
        'section': ParagraphStyle('section', fontName=FONT, fontSize=13, leading=19, textColor=colors.HexColor('#1e3a8a'), spaceBefore=8, spaceAfter=6, keepWithNext=True),
        'subsection': ParagraphStyle('subsection', fontName=FONT, fontSize=11.5, leading=17, textColor=colors.HexColor('#1e3a8a'), spaceBefore=6, spaceAfter=5, keepWithNext=True),
        'caption': ParagraphStyle('caption', fontName=FONT, fontSize=8.5, leading=12, textColor=colors.HexColor('#475569'), spaceBefore=5, spaceAfter=8, wordWrap='CJK'),
        'cell': ParagraphStyle('cell', fontName=FONT, fontSize=8.4, leading=13, wordWrap='CJK'),
    }
    image_plan = []
    story = []
    lines = MD.read_text(encoding='utf-8').splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if line == '<!-- pagebreak -->':
            story.append(PageBreak())
            continue
        match = re.fullmatch(r'!\[(.*?)\]\((.*?)\)', line)
        if match:
            caption, relative = match.groups()
            path = ROOT / relative
            with PILImage.open(path) as source:
                w, h = source.size
                source.verify()
            max_height = 95 * mm if path.name.startswith(('csdi_', 'imdiffusion_')) else 90 * mm
            scale = min(WIDTH / w, max_height / h)
            display_w, display_h = w * scale, h * scale
            assert abs(display_w / display_h - w / h) < 1e-10
            image_plan.append({'file': str(path), 'pixels': [w, h],
                               'display_points': [display_w, display_h],
                               'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            pic = Image(str(path), width=display_w, height=display_h)
            pic.hAlign = 'CENTER'
            story.append(KeepTogether([pic, Paragraph(inline(caption), styles['caption'])]))
            continue
        if line.startswith('|'):
            block = [line]
            while i < len(lines) and lines[i].strip().startswith('|'):
                block.append(lines[i].strip())
                i += 1
            rows = [[c.strip() for c in row.strip('|').split('|')] for row in block]
            rows = [rows[0]] + rows[2:]
            cells = [[Paragraph(inline(c), styles['cell']) for c in row] for row in rows]
            table = Table(cells, colWidths=[WIDTH * f for f in [.15, .20, .26, .21, .18]], repeatRows=1, hAlign='CENTER')
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dbeafe')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
                ('GRID', (0, 0), (-1, -1), .4, colors.HexColor('#cbd5e1')),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 7),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ]))
            story.append(table)
            continue
        if line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            key = {1: 'title', 2: 'subtitle', 3: 'section', 4: 'subsection'}[level]
            story.append(Paragraph(inline(line[level:].strip()), styles[key]))
            continue
        key = 'caption' if line.startswith('*2:8') or line.startswith('结果来源：') else 'body'
        story.append(Paragraph(inline(line), styles[key]))

    def furniture(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#cbd5e1'))
        canvas.line(15 * mm, 13 * mm, PAGE_W - 15 * mm, 13 * mm)
        canvas.setFont(FONT, 8)
        canvas.setFillColor(colors.HexColor('#64748b'))
        footer = '绝缘子检测算法进展'
        canvas.drawString(15 * mm, 9 * mm, footer)
        canvas.setFont('DejaVu', 8)
        canvas.drawString(15 * mm + pdfmetrics.stringWidth(footer, FONT, 8), 9 * mm, ' | v4')
        canvas.drawRightString(PAGE_W - 15 * mm, 9 * mm, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(str(PDF), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=14 * mm, bottomMargin=18 * mm,
                            title='绝缘子检测算法进展（v4版本）', author='v4算法实验报告',
                            subject='异常标注优化与扩散数据增强实验', pageCompression=1)
    doc.build(story, onFirstPage=furniture, onLaterPages=furniture)
    # Inspect real PDF objects, not just the requested layout geometry.
    geometry = ASSETS / 'pdf_geometry.xml'
    subprocess.run(['pdftohtml', '-xml', '-hidden', '-zoom', '1', '-i', str(PDF), str(geometry)],
                   check=True, capture_output=True)
    tree = ET.parse(geometry)
    pages = tree.findall('page')
    if len(pages) != 4:
        raise AssertionError(f'Expected four deliberately laid-out pages; got {len(pages)}')
    # -i omits images, so also obtain image positions without that flag.
    subprocess.run(['pdftohtml', '-xml', '-hidden', '-zoom', '1', str(PDF), str(ASSETS / 'pdf_images.xml')],
                   check=True, capture_output=True)
    image_tree = ET.parse(ASSETS / 'pdf_images.xml')
    actual_images = []
    for page in image_tree.findall('page'):
        for node in page.findall('image'):
            top, left, width, height = [float(node.attrib[k]) for k in ['top', 'left', 'width', 'height']]
            assert 0 <= left and left + width <= float(page.attrib['width']) + 1
            assert 0 <= top and top + height <= float(page.attrib['height']) - 35
            actual_images.append((page.attrib['number'], width, height))
        for node in page.findall('text'):
            if (node.text or '').strip():
                left, top = float(node.attrib['left']), float(node.attrib['top'])
                assert left >= 0 and top >= 0
                assert left + float(node.attrib['width']) <= float(page.attrib['width']) + 1
                assert top + float(node.attrib['height']) <= float(page.attrib['height']) + 1
    assert len(actual_images) == len(image_plan) == 5, actual_images
    for actual, planned in zip(actual_images, image_plan):
        _, width, height = actual
        pw, ph = planned['pixels']
        assert abs(width / height / (pw / ph) - 1) < .015, (actual, planned)
        planned['page'] = int(actual[0])
        planned['actual_display_points'] = [width, height]
    text = subprocess.check_output(['pdftotext', str(PDF), '-'], text=True)
    for expected in ['一、摘要', '二、主要进展', '三、问题', '四、Todo', '0.2298', '0.2786', '10.67%', '图5']:
        assert expected in text, f'Missing PDF content: {expected}'
    assert '\ufffd' not in text
    assert 'CSDI' not in text, 'Obsolete CSDI explanation remained in the updated report'
    assert 'IMDiffusion' in text
    assert all('imdiffusion_' in item['file'] for item in image_plan[2:])
    subprocess.run(['pdftoppm', '-scale-to', '1600', '-png', str(PDF), str(ASSETS / 'page')], check=True)
    previews = sorted(ASSETS.glob('page-*.png'))
    assert len(previews) == 4
    thumbs = []
    for preview in previews:
        with PILImage.open(preview) as im:
            thumb = im.convert('RGB')
            thumb.thumbnail((650, 920))
            thumbs.append(thumb)
    contact = PILImage.new('RGB', (1340, 1900), '#e2e8f0')
    for index, thumb in enumerate(thumbs):
        contact.paste(thumb, (10 + index % 2 * 670, 10 + index // 2 * 950))
    contact.save(ASSETS / 'page_contact_sheet.png')
    manifest = {'pages': len(pages), 'embedded_images': image_plan, 'pdf': str(PDF),
                'pdf_sha256': hashlib.sha256(PDF.read_bytes()).hexdigest(),
                'checks': ['4 A4 pages', '5 complete aspect-preserving images', 'image and text bounds',
                           'expected Chinese text and results', 'rendered all pages'],
                'fonts': ['embedded DroidSansFallback', 'embedded DejaVuSans'],
                'source_markdown': str(MD)}
    (ASSETS / 'validation.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    shutil.copy2(PDF, REPORTS / PDF.name)
    (REPORTS / MD.name).write_text(MD.read_text(encoding='utf-8').replace(
        'v4_results_20261009/06_reports/V4_PROGRESS_ASSETS/', 'V4_PROGRESS_ASSETS/'), encoding='utf-8')
    print('PDF:', PDF)
    print('Pages:', len(pages), 'Figures:', len(image_plan), 'Bytes:', PDF.stat().st_size)
    print('Preview:', ASSETS / 'page_contact_sheet.png')


if __name__ == '__main__':
    main()
