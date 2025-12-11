# Används som
#
#   python3 extract_segment.py <PDF-FILNAMN>
#
# där PDF-FILNAMN är en av pdf-filerna från
#
#   Språkbanken Text (2020). Svenska tidningar 1818-1870 (uppdaterad:
#   2020-05-26). [Data set]. Språkbanken Text.
#   https://doi.org/10.23695/9bnq-xc71
#
# eller
#
#   Språkbanken Text (2022). Svenska tidningar 1871-1906 (uppdaterad:
#   2022-05-03). [Data set]. Språkbanken Text.
#   https://doi.org/10.23695/6kg4-8h62
#
# Segmenten i pdf:erna sparas som png filer i arbetsmappen.
#
# Skriptet spottar ur sig lite info om pdf:en och de extraherade
# bilderna på stdout (som man kan ignorera så länge allting verkar
# funka).
#

import sys
import math
from pathlib import Path
import pymupdf # ...pip install pymupdf


def extract_segments(fn):
    doc = pymupdf.Document(fn)
    page = doc.load_page(0)

    # Segmenten anges med rektanglar som ritats över bilden i pdf:en. Varje segment omges av tre rektanglar, vi väljer den mittersta som lagom... 
    segment_rects = [path['items'][0][1] for path in page.get_drawings()[1:-1:3]] 
    print(*enumerate(segment_rects,start=1), sep='\n')
    print(page.rect) # hela sidan
    
    img_info = page.get_images()[0]
    img = pymupdf.Pixmap(doc, img_info[0])
    # print(img)

    h_scale = img.height / page.rect.height
    w_scale = img.width / page.rect.width
    assert math.isclose(h_scale, w_scale, rel_tol=.0001)
    
    # print(w_scale, h_scale)
    
    scale_matrix = pymupdf.Matrix(h_scale, w_scale)
    
    for i, segment_rect in enumerate(segment_rects, start=1):
        print(i)    
        print(segment_rect)
        segment_rect *= scale_matrix 
        print(segment_rect)
        segment_irect = pymupdf.IRect(segment_rect)+(0,0,1,1)
        print(segment_irect)
        segment = pymupdf.Pixmap(img, img.width, img.height, segment_irect)
        segment.save(Path(fn).stem+'-%04d.png' % i)
        print()

    

if __name__=="__main__":
    extract_segments(sys.argv[1])
