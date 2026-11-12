"""Entrada separada. No modifica web.py ni el lanzador habitual."""
import multiprocessing as mp
from experimental.web_emociones import main

if __name__ == '__main__':
    mp.freeze_support()
    main()
