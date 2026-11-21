"""Cut the reel the intro television plays: public/greatest_moments.mp4 (+ .vtt with one label per clip).

Source: "The Beauty of Football - Greatest Moments" by LeBreton (https://www.youtube.com/watch?v=A-r9sj7zHJ4, 12:22). Every classic moment
in it is kept, from the black-and-white legends to the 2019 comebacks, each trimmed to its key action; what is not football (the opening
quote card, the interview shots, the stadium aerials, the closing "share / like / comment" card) is left out. Clips stay in the source's
order so the music flows, and are joined with short crossfades. Every frame is cropped to 16:9 inside the letterbox, which also drops the
burnt-in caption band and the channel watermark; the labels come back as WebVTT cues that the television shows on its screen.

Needs yt-dlp and ffmpeg on PATH. The download and the intermediate clips go to data/raw/ (git-ignored) and are reused on later runs.

    uv run python scripts/make_greatest_moments.py
"""

import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.youtube.com/watch?v=A-r9sj7zHJ4"
SRC = ROOT / "data" / "raw" / "beauty_of_football.mp4"
TMP = ROOT / "data" / "raw" / "greatest_moments"
OUT = ROOT / "public" / "greatest_moments.mp4"
VTT = OUT.with_suffix(".vtt")

# (start s, end s, label) in the source video, in its order. Labels follow the source's own captions.
CLIPS = [
    # The opening portraits and farewells move fast: about a second per person.
    (12.0, 13.2, "Ferenc Puskás"),
    (15.5, 16.7, "Pelé"),
    (17.6, 18.8, "Bobby Charlton"),
    (19.6, 20.8, "Franz Beckenbauer"),
    (21.4, 22.6, "Johan Cruyff"),
    (23.0, 24.2, "Diego Maradona"),
    (34.6, 35.8, "Zidane's farewell"),
    (37.6, 38.8, "Del Piero's farewell"),
    (40.4, 41.6, "Oliver Kahn's farewell"),
    (42.7, 43.6, "David Beckham's farewell"),
    (44.8, 46.0, "Schweinsteiger's farewell"),
    (47.6, 48.8, "Iniesta's farewell"),
    (51.4, 52.6, "Totti's farewell"),
    (54.0, 55.2, "Gerrard's farewell"),
    (59.0, 65.5, "Iniesta's winner · 2010 World Cup final"),
    (67.4, 71.1, "Wiltord's last-minute equaliser · Euro 2000 final"),
    (71.6, 76.5, "Trezeguet's golden goal · Euro 2000 final"),
    (79.9, 86.8, "Rooney's bicycle kick v Manchester City · 2011"),
    (90.0, 96.5, "Messi v Athletic Club · 2013"),
    (106.5, 112.8, "Ibrahimović's bicycle kick v England · 2012"),
    (119.0, 126.5, "Robben v Spain · 2014 World Cup (5-1)"),
    (130.3, 131.8, "The Cruyff turn · 1974"),
    (132.1, 137.9, "Neymar v Villarreal · 2015"),
    (138.3, 143.1, "Bergkamp v Newcastle · 2002"),
    (143.6, 149.1, "Ronaldo v Barcelona · 2017 Supercopa"),
    (149.6, 157.0, "Giggs v Arsenal · 1999 FA Cup semi-final"),
    (158.0, 165.2, "Bale's bicycle kick · 2018 Champions League final"),
    (165.6, 173.5, "Milito v Bayern · 2010 Champions League final"),
    (174.9, 182.0, "Pelé"),
    (182.6, 187.4, "Pelé v Italy · 1970 World Cup final"),
    (187.9, 192.7, "Zidane v Real Betis · 1995"),
    (193.1, 198.7, "Roberto Carlos's free kick v France · 1997"),
    (199.1, 202.7, "Ernie Hunt's free kick v Everton · 1970"),
    (203.1, 207.7, "Juninho's free kick v Barcelona · 2009"),
    (208.1, 212.2, "Juninho's free kick v Real Madrid · 2005"),
    (214.0, 225.5, "Messi v Athletic Club · 2015 Copa del Rey final"),
    (234.5, 241.4, "Salah v Everton · 2017"),
    (244.0, 251.5, "Beckham's free kick v Greece · 2001"),
    (255.0, 262.5, "Götze, 113th minute · 2014 World Cup final"),
    (267.5, 275.5, "Agüero, 93:20 · Manchester City's title, 2012"),
    (278.6, 282.9, "Messi v Bayern · 2015"),
    (283.3, 285.9, "Maradona"),
    (286.3, 291.9, "Pavard v Argentina · 2018 World Cup"),
    (293.5, 300.5, "Özil v Ludogorets · 2016"),
    (302.5, 307.5, "James Rodríguez v Uruguay · 2014 World Cup"),
    (312.0, 320.5, "Ronaldo's bicycle kick v Juventus · 2018"),
    (321.3, 323.2, "Bobby Charlton"),
    (323.5, 328.0, "Maradona's \"Hand of God\" · 1986 World Cup"),
    (328.3, 335.8, "Leicester City, champions · 2016"),
    (336.1, 339.2, "Panenka · Euro 1976 final"),
    (340.5, 349.5, "Bergkamp v Argentina · 1998 World Cup"),
    (353.5, 359.5, "Chelsea's late equaliser · 2012 Champions League final"),
    (362.0, 366.6, "Chelsea win on penalties · 2012 Champions League final"),
    (404.6, 413.2, "The fans"),
    (433.3, 445.0, "The goal of the century · Maradona 1986, Messi 2007"),
    (450.0, 456.0, "Ronaldinho, applauded at the Bernabéu · 2005"),
    (459.6, 462.5, "Gerrard"),
    (463.3, 466.0, "Ronaldo"),
    (466.6, 469.5, "Zidane's volley · 2002 Champions League final"),
    (472.0, 477.7, "Roma 3-0 Barcelona · 2018 Champions League"),
    (478.1, 481.0, "Neymar v Flamengo · 2011"),
    (481.3, 485.8, "Greece, 150-1 outsiders, win Euro 2004"),
    (486.1, 488.0, "Ronaldo"),
    (488.3, 495.0, "Robben's late winner · 2013 Champions League final"),
    (498.0, 505.0, "Liverpool, 3-0 down at half-time · 2005 Champions League final"),
    (508.6, 514.5, "Liverpool win on penalties · 2005 Champions League final"),
    (517.3, 523.0, "Garrincha"),
    (523.3, 525.8, "Ferenc Puskás"),
    (526.0, 532.5, "Zidane v Brazil · 2006 World Cup"),
    (535.0, 541.6, "Van Persie's flying header v Spain · 2014 World Cup"),
    (559.0, 568.2, "Lewandowski, five goals in nine minutes · 2015"),
    (568.6, 575.2, "Thuram's only two goals for France · 1998 World Cup"),
    (577.5, 584.4, "Barcelona 6-1 PSG · 2017, the biggest comeback"),
    (587.0, 594.5, "Tottenham, 0-3 down on aggregate at Ajax · 2019"),
    (608.5, 620.5, "Lucas Moura's hat-trick, 96th-minute winner · 2019"),
    (629.5, 639.5, "Germany 7-1 Brazil · 2014 World Cup"),
    (653.5, 663.5, "Liverpool 4-0 Barcelona · 2019, the corner taken quickly"),
    (670.0, 674.6, "Liverpool 4-0 Barcelona · 2019"),
    (675.3, 682.2, "France 3-0 Brazil · 1998 World Cup final"),
    (689.0, 695.4, "France 4-2 Croatia · 2018 World Cup final"),
    (696.0, 704.6, "Champions"),
    (709.0, 719.7, "Legends"),
]
CROP = "crop=1074:604:166:58"  # 16:9 picture inside the 1280x720 letterbox, left of the watermark
SIZE = "640:360"
FADE = 0.25  # crossfade between clips (s)


def stamp(t: float) -> str:
    return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{t % 60:06.3f}"


def cut(i: int, a: float, b: float) -> Path:
    """One clip, cropped and scaled, as a near-lossless intermediate."""
    out = TMP / f"{i:03d}_{a:.1f}_{b:.1f}.mp4"
    if not out.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a}", "-t", f"{b - a}", "-i", str(SRC),
                        "-vf", f"{CROP},scale={SIZE}:flags=lanczos,setsar=1,fps=24,format=yuv420p",
                        "-af", "aformat=sample_rates=48000:channel_layouts=stereo",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "12", "-c:a", "pcm_s16le", "-f", "mov", str(out)],
                       check=True, stdin=subprocess.DEVNULL)
    return out


def main() -> None:
    for tool in ("ffmpeg", "yt-dlp"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found on PATH")
    if not SRC.exists():
        SRC.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL} ...", flush=True)
        subprocess.run(["yt-dlp", "-q", "--no-playlist", "-f", "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720]",
                        "--merge-output-format", "mp4", "-o", str(SRC), URL], check=True)

    TMP.mkdir(parents=True, exist_ok=True)
    print(f"cutting {len(CLIPS)} clips ...", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        parts = list(pool.map(lambda c: cut(c[0], *c[1][:2]), enumerate(CLIPS)))

    # Chain the crossfades; each one starts FADE seconds before the running total ends.
    inputs = [arg for p in parts for arg in ("-i", str(p))]
    chains, cues = [f"[0:v]settb=AVTB[v0]", "[0:a]asetpts=PTS-STARTPTS[a0]"], [(0.0, CLIPS[0][2])]
    vlast, alast, total = "v0", "a0", CLIPS[0][1] - CLIPS[0][0]
    for i in range(1, len(CLIPS)):
        offset = total - FADE
        chains.append(f"[{i}:v]settb=AVTB[v{i}];[{vlast}][v{i}]xfade=transition=fade:duration={FADE}:offset={offset:.3f}[vx{i}];"
                      f"[{i}:a]asetpts=PTS-STARTPTS[a{i}];[{alast}][a{i}]acrossfade=d={FADE}[ax{i}]")
        cues.append((offset + FADE / 2, CLIPS[i][2]))
        vlast, alast, total = f"vx{i}", f"ax{i}", offset + (CLIPS[i][1] - CLIPS[i][0])
    chains.append(f"[{vlast}]fade=t=in:d=0.4,fade=t=out:st={total - 1.0:.3f}:d=1.0[vout];"
                  f"[{alast}]afade=t=in:d=0.4,afade=t=out:st={total - 1.5:.3f}:d=1.5[aout]")

    OUT.parent.mkdir(exist_ok=True)
    print(f"joining them into {OUT.relative_to(ROOT)} ({total / 60:.1f} min) ...", flush=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(chains), "-map", "[vout]", "-map", "[aout]",
                    "-c:v", "libx264", "-preset", "slow", "-crf", "32", "-profile:v", "high", "-pix_fmt", "yuv420p", "-g", "48",
                    "-c:a", "aac", "-b:a", "80k", "-movflags", "+faststart", str(OUT)], check=True, stdin=subprocess.DEVNULL)

    ends = [t for t, _ in cues[1:]] + [total]
    VTT.write_text("WEBVTT\n\n" + "\n".join(f"{stamp(t)} --> {stamp(e)}\n{label}\n" for (t, label), e in zip(cues, ends)), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB) and {VTT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
