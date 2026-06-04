#!/usr/bin/env python3
"""
Faithful PIL preview of the deck, using the SAME inch-geometry as build_deck.render().
Lets us visually QA composition without LibreOffice. Renders defense_build/preview/NN.png.
Run:  python3 defense_build/preview.py
"""
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib
from build_deck import SLIDES, resolve, SW, SH

PX = 100
W, H = int(SW*PX), int(SH*PX)
TTF = os.path.join(matplotlib.get_data_path(), "fonts/ttf")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preview")
os.makedirs(OUT, exist_ok=True)

NAVY=(0x1A,0x2A,0x4F); TEAL=(0x1C,0x72,0x93); ACCENT=(0xB5,0x65,0x1D)
GREY=(0x8A,0x94,0xA6); DARK=(0x22,0x2A,0x36); LIGHT=(0xEA,0xF1,0xF5); WHITE=(255,255,255)

def font(pt, bold=False, italic=False):
    name = "DejaVuSans"
    if bold: name += "-Bold"
    elif italic: name += "-Oblique"
    return ImageFont.truetype(os.path.join(TTF, name+".ttf"), int(round(pt*PX/72)))

def wrap(draw, text, f, maxw):
    out=[]
    for para in text.split("\n"):
        words=para.split(" "); line=""
        for w in words:
            t=(line+" "+w).strip()
            if draw.textlength(t, font=f) <= maxw or not line:
                line=t
            else:
                out.append(line); line=w
        out.append(line)
    return out

def text(draw, s, box, pt, color, bold=False, italic=False, align="l", valign="t", ls=1.12):
    l,t,w,h=[v*PX for v in box]
    f=font(pt,bold,italic)
    lines=wrap(draw,s,f,w)
    lh=(f.getbbox("Ag")[3]-f.getbbox("Ag")[1])*ls + pt*PX/72*0.35
    total=lh*len(lines)
    y = t + (h-total)/2 if valign=="m" else t
    for ln in lines:
        lw=draw.textlength(ln,font=f)
        x = l + (w-lw)/2 if align=="c" else (l+w-lw if align=="r" else l)
        draw.text((x,y), ln, font=f, fill=color)
        y+=lh

def box_rect(draw, box, fill):
    l,t,w,h=[v*PX for v in box]; draw.rectangle([l,t,l+w,t+h], fill=fill)

def fitpaste(canvas, path, box):
    if not path: return
    l,t,w,h=[v*PX for v in box]
    im=Image.open(path).convert("RGBA")
    iw,ih=im.size; ar=iw/ih; bar=w/h
    nw,nh=(w,w/ar) if ar>bar else (h*ar,h)
    im=im.resize((max(1,int(nw)),max(1,int(nh))))
    x=int(l+(w-nw)/2); y=int(t+(h-nh)/2)
    canvas.paste(im,(x,y),im)

def titlebar(c,d,title):
    box_rect(d,(0,0,SW,1.02),NAVY); box_rect(d,(0,1.02,SW,0.06),ACCENT)
    text(d,title,(0.55,0,SW-1.1,1.02),26,WHITE,bold=True,valign="m")

def footer(d,n):
    text(d,"ARSPI-Net   ·   A. Lane",(0.5,7.16,5,0.3),9.5,GREY)
    text(d,str(n),(SW-1.1,7.16,0.6,0.3),9.5,GREY,align="r")

def bullets(c,d,items,box,size=19):
    l,t,w,h=box; f=font(size);
    # measure
    lines=[]
    for it in items: lines+=wrap(d,"▸  "+it,f,w*PX)
    lh=(f.getbbox("Ag")[3]-f.getbbox("Ag")[1])*1.15+size*PX/72*0.55
    y=t*PX+(h*PX-lh*len(lines))/2
    for it in items:
        for j,ln in enumerate(wrap(d,"▸  "+it,f,w*PX)):
            d.text((l*PX,y),ln,font=f,fill=DARK); y+=lh
        y+=size*PX/72*0.25

def render(sl,n):
    c=Image.new("RGB",(W,H),WHITE); d=ImageDraw.Draw(c)
    k=sl["kind"]
    if k=="title":
        box_rect(d,(0,0,SW,SH),NAVY); box_rect(d,(0,3.05,SW,0.05),ACCENT)
        text(d,sl["title"],(0.8,1.4,SW-1.6,1.4),60,WHITE,bold=True,align="c",valign="m")
        text(d,sl["sub"],(0.8,3.2,SW-1.6,1.4),23,LIGHT,align="c")
        text(d,sl["foot"],(0.8,5.7,SW-1.6,1.2),17,GREY,align="c")
    elif k=="section":
        box_rect(d,(0,0,SW,SH),NAVY)
        if sl.get("kicker"): text(d,sl["kicker"],(0.9,2.7,SW-1.8,0.6),18,ACCENT,bold=True,align="c")
        text(d,sl["title"],(0.9,3.2,SW-1.8,1.4),40,WHITE,bold=True,align="c",valign="m")
    elif k=="statement":
        box_rect(d,(0,0,0.28,SH),ACCENT)
        if sl.get("kicker"): text(d,sl["kicker"],(1.0,1.5,SW-2,0.6),18,TEAL,bold=True)
        text(d,sl["title"],(1.0,2.0,SW-2,3.2),36,NAVY,bold=True,valign="m")
        if sl.get("sub"): text(d,sl["sub"],(1.0,5.2,SW-2,1.3),20,DARK,italic=True)
        footer(d,n)
    elif k=="close":
        box_rect(d,(0,0,SW,SH),NAVY)
        text(d,sl["title"],(0.8,2.4,SW-1.6,1.6),48,WHITE,bold=True,align="c",valign="m")
        text(d,sl["sub"],(0.8,4.2,SW-1.6,1.6),20,LIGHT,align="c")
    else:
        titlebar(c,d,sl["title"]); top=1.32
        if sl.get("headline"):
            text(d,sl["headline"],(0.55,top,SW-1.1,0.8),21,ACCENT,bold=True,valign="m"); top+=0.92
        bottom=7.05; area=(0.55,top,SW-1.1,bottom-top)
        def cap(box,s):
            if s: text(d,s,(box[0],box[1]+box[3]-0.3,box[2],0.3),11.5,GREY,italic=True,align="c")
        if k=="figure":
            b=(0.55,top,SW-1.1,(bottom-top)-(0.3 if sl.get('caption') else 0)); fitpaste(c,resolve(sl["img"]),b); cap(area,sl.get("caption"))
        elif k=="figtext":
            fw=(SW-1.1)*0.58; fitpaste(c,resolve(sl["img"]),(0.55,top,fw,(bottom-top)-0.3)); cap((0.55,top,fw,bottom-top),sl.get("caption"))
            bullets(c,d,sl["bullets"],(0.55+fw+0.3,top,(SW-1.1)-fw-0.3,bottom-top),sl.get("bsize",19))
        elif k=="eqfig":
            eqh=1.25; fitpaste(c,resolve(sl["img"]),(0.55,top,SW-1.1,(bottom-top)-eqh-(0.3 if sl.get('caption') else 0)))
            cap((0.55,top,SW-1.1,(bottom-top)-eqh),sl.get("caption"))
            if sl.get("eq"): fitpaste(c,resolve(sl["eq"]),(1.8,bottom-eqh,SW-3.6,eqh-0.05))
        elif k=="eq":
            eqh=1.7; fitpaste(c,resolve(sl["eq"]),(1.4,top+0.1,SW-2.8,eqh))
            if sl.get("bullets"): bullets(c,d,sl["bullets"],(1.2,top+eqh+0.25,SW-2.4,bottom-(top+eqh+0.25)),sl.get("bsize",19))
        elif k=="twofig":
            gap=0.4; cw=(SW-1.1-gap)/2
            fitpaste(c,resolve(sl["img"]),(0.55,top,cw,(bottom-top)-0.3)); cap((0.55,top,cw,bottom-top),sl.get("caption"))
            fitpaste(c,resolve(sl["img2"]),(0.55+cw+gap,top,cw,(bottom-top)-0.3)); cap((0.55+cw+gap,top,cw,bottom-top),sl.get("caption2"))
        elif k=="table":
            fitpaste(c,resolve(sl["img"]),(0.9,top,SW-1.8,bottom-top))
        elif k=="tabletext":
            tw=(SW-1.1)*0.6; fitpaste(c,resolve(sl["img"]),(0.55,top,tw,bottom-top))
            bullets(c,d,sl["bullets"],(0.55+tw+0.3,top,(SW-1.1)-tw-0.3,bottom-top),sl.get("bsize",18))
        elif k=="bullets":
            bullets(c,d,sl["bullets"],(0.8,top,SW-1.6,bottom-top),sl.get("bsize",22))
        elif k=="movie":
            poster=resolve(sl["poster"]) if not sl["poster"].endswith(".pdf") else None
            from build_deck import pdf_to_png
            poster=poster or pdf_to_png(sl["poster"])
            fitpaste(c,poster,(0.55,top,(SW-1.1)*0.62,bottom-top))
            if sl.get("bullets"):
                bx=0.55+(SW-1.1)*0.62+0.3; bullets(c,d,sl["bullets"],(bx,top,(SW-0.55)-bx,bottom-top),18)
        footer(d,n)
    return c

def main():
    n=0
    for i,sl in enumerate(SLIDES):
        if sl["kind"] not in ("title","section","close"): n+=1
        c=render(sl,n); c.save(os.path.join(OUT,f"{i:02d}_{sl['kind']}.png"))
    print("preview slides written:", len(SLIDES), "->", OUT)

if __name__=="__main__":
    main()
