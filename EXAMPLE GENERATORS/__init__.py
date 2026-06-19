"""Synthetic data generators for each problem type in the competition."""

from .base import BaseGenerator
from .bit_manipulation import BitManipulationGenerator
from .encryption import EncryptionGenerator
from .number_conversion import NumberConversionGenerator
from .unit_conversion import UnitConversionGenerator
from .gravitational import GravitationalGenerator
from .transformation import TransformationGenerator

GENERATORS = {
    "bit_manipulation": BitManipulationGenerator,
    "encryption": EncryptionGenerator,
    "number_conversion": NumberConversionGenerator,
    "unit_conversion": UnitConversionGenerator,
    "gravitational": GravitationalGenerator,
    "transformation": TransformationGenerator,
}

__all__ = [
    "BaseGenerator",
    "GENERATORS",
    "BitManipulationGenerator",
    "EncryptionGenerator",
    "NumberConversionGenerator",
    "UnitConversionGenerator",
    "GravitationalGenerator",
    "TransformationGenerator",
]
