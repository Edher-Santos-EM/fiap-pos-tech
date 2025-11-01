"""Barras de progresso elegantes usando tqdm."""

from tqdm import tqdm
from typing import Optional


def create_progress_bar(
    total: int,
    desc: str,
    unit: str = "it",
    colour: str = "green"
) -> tqdm:
    """
    Cria barra de progresso elegante.

    Args:
        total: Total de items
        desc: Descrição
        unit: Unidade (frame, cena, etc.)
        colour: Cor da barra

    Returns:
        Objeto tqdm
    """
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        colour=colour,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {rate_fmt}'
    )
