import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import json
import hashlib
from dataclasses import dataclass, field, asdict
from urllib.parse import urljoin, urlparse
import time
import logging
import tempfile

# Опциональный импорт Selenium
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# ==================== PLOTLY HELPER ====================
def _apply_transparent_bg(fig):
    """Apply transparent background to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2c3e50')
    )
    return fig

# ==================== НАСТРОЙКА ЛОГИРОВАНИЯ ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

COLLECTOR_AVAILABLE = True
WEB_SCRAPERS_AVAILABLE = True


# ==================== КЛАССЫ ДАННЫХ (data_collector) ====================

@dataclass
class UAVModelRaw:
    model_name: str
    manufacturer_name: str
    country_name: Optional[str] = None
    type_name: Optional[str] = None
    status_name: Optional[str] = "Активная"
    payload_capacity_kg: Optional[float] = None
    takeoff_mass_kg: Optional[float] = None
    flight_time_min: Optional[float] = None
    max_speed_kmh: Optional[float] = None
    max_range_km: Optional[float] = None
    max_altitude_m: Optional[float] = None
    wingspan_or_rotor_m: Optional[float] = None
    length_m: Optional[float] = None
    battery_capacity_kwh: Optional[float] = None
    fuel_consumption_l_per_h: Optional[float] = None
    wind_resistance_ms: Optional[float] = None
    power_type: Optional[str] = None
    ip_rating: Optional[str] = None
    temp_range_min_c: Optional[float] = None
    temp_range_max_c: Optional[float] = None
    labor_cost_per_ha: Optional[float] = None
    energy_cost_per_ha_min: Optional[float] = None
    energy_cost_per_ha_max: Optional[float] = None
    recommended_field_area_min_ha: Optional[float] = None
    recommended_field_area_max_ha: Optional[float] = None
    source_url: Optional[str] = None
    notes: Optional[str] = None
    applications: List[Dict] = field(default_factory=list)
    agro_zones: List[Dict] = field(default_factory=list)
    operations: List[Dict] = field(default_factory=list)
    sensors: List[Dict] = field(default_factory=list)
    prices: List[Dict] = field(default_factory=list)
    tco_entries: List[Dict] = field(default_factory=list)


class DataValidator:
    TYPE_ALIASES = {
        'мультикоптер': 'Мультиротор', 'multicopter': 'Мультиротор', 'multirotor': 'Мультиротор',
        'дрон': 'Мультиротор', 'самолет': 'Самолётный', 'fixed-wing': 'Самолётный',
        'fixed wing': 'Самолётный', 'вертолет': 'Вертолётный', 'helicopter': 'Вертолётный',
        'rotary': 'Вертолётный', 'vtol': 'Гибридный (VTOL)', 'гибрид': 'Гибридный (VTOL)',
    }
    POWER_ALIASES = {
        'электро': 'Электрический', 'electric': 'Электрический', 'бензин': 'Бензиновый',
        'gasoline': 'Бензиновый', 'дизель': 'Дизельный', 'diesel': 'Дизельный',
        'гибрид': 'Гибридный', 'hybrid': 'Гибридный',
    }

    @classmethod
    def normalize_type(cls, raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        raw_lower = raw.lower().strip()
        for alias, canonical in cls.TYPE_ALIASES.items():
            if alias in raw_lower:
                return canonical
        return raw.strip()

    @classmethod
    def normalize_power(cls, raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        raw_lower = raw.lower().strip()
        for alias, canonical in cls.POWER_ALIASES.items():
            if alias in raw_lower:
                return canonical
        return raw.strip()

    @classmethod
    def validate_model(cls, model: UAVModelRaw) -> Tuple[bool, List[str]]:
        errors = []
        if not model.model_name:
            errors.append("Отсутствует обязательное поле: model_name")
        if not model.manufacturer_name:
            errors.append("Отсутствует обязательное поле: manufacturer_name")
        if model.payload_capacity_kg is not None and model.payload_capacity_kg < 0:
            errors.append("Грузоподъёмность не может быть отрицательной")
        if model.flight_time_min is not None and model.flight_time_min > 1440:
            errors.append("Время полёта > 24ч — подозрительное значение")
        return len(errors) == 0, errors

    @classmethod
    def clean_numeric(cls, val: Any) -> Optional[float]:
        if val is None or pd.isna(val):
            return None
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            match = re.search(r"[-+]?\d*[\.,]?\d+", val.replace(' ', '').replace(',', '.'))
            if match:
                try:
                    return float(match.group().replace(',', '.'))
                except ValueError:
                    return None
        return None

    @classmethod
    def normalize_model(cls, model: UAVModelRaw) -> UAVModelRaw:
        model.type_name = cls.normalize_type(model.type_name)
        model.power_type = cls.normalize_power(model.power_type)
        numeric_fields = [
            'payload_capacity_kg', 'takeoff_mass_kg', 'flight_time_min', 'max_speed_kmh',
            'max_range_km', 'max_altitude_m', 'wingspan_or_rotor_m', 'length_m',
            'battery_capacity_kwh', 'fuel_consumption_l_per_h', 'wind_resistance_ms',
            'labor_cost_per_ha', 'energy_cost_per_ha_min', 'energy_cost_per_ha_max',
            'recommended_field_area_min_ha', 'recommended_field_area_max_ha',
            'temp_range_min_c', 'temp_range_max_c'
        ]
        for f in numeric_fields:
            setattr(model, f, cls.clean_numeric(getattr(model, f)))
        for attr in ['model_name', 'manufacturer_name', 'country_name', 'ip_rating']:
            val = getattr(model, attr)
            if val and isinstance(val, str):
                setattr(model, attr, val.strip())
        return model


class DatabaseIngestor:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_logs_table()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _ensure_logs_table(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS collection_logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_name TEXT, url TEXT, model_name TEXT, manufacturer_name TEXT,
                    status TEXT, message TEXT, raw_data_json TEXT,
                    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def _get_or_create(self, conn: sqlite3.Connection, table: str,
                       id_col: str, name_col: str, name_val: str,
                       extra_cols: Optional[Dict] = None) -> int:
        cur = conn.execute(f"SELECT {id_col} FROM {table} WHERE {name_col} = ?", (name_val,))
        row = cur.fetchone()
        if row:
            return row[0]
        cols = [name_col]
        vals = [name_val]
        if extra_cols:
            for c, v in extra_cols.items():
                cols.append(c)
                vals.append(v)
        placeholders = ', '.join(['?'] * len(vals))
        sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
        cur = conn.execute(sql, vals)
        conn.commit()
        return cur.lastrowid

    def _model_exists(self, conn: sqlite3.Connection, model_name: str, manufacturer_id: int) -> Optional[int]:
        cur = conn.execute(
            "SELECT model_id FROM bpla_models WHERE model_name = ? AND manufacturer_id = ?",
            (model_name, manufacturer_id)
        )
        row = cur.fetchone()
        return row[0] if row else None

    def ingest(self, model: UAVModelRaw, source_name: str = "manual",
               update_existing: bool = False) -> Tuple[bool, str]:
        model = DataValidator.normalize_model(model)
        is_valid, errors = DataValidator.validate_model(model)
        if not is_valid:
            self._log(source_name, model, 'validation_error', '; '.join(errors))
            return False, f"Ошибка валидации: {'; '.join(errors)}"

        try:
            with self._connect() as conn:
                country_id = None
                if model.country_name:
                    country_id = self._get_or_create(conn, 'countries', 'country_id', 'country_name',
                                                     model.country_name)
                man_extra = {'country_id': country_id} if country_id else {}
                manufacturer_id = self._get_or_create(
                    conn, 'manufacturers', 'manufacturer_id', 'manufacturer_name',
                    model.manufacturer_name, man_extra
                )
                type_id = None
                if model.type_name:
                    type_id = self._get_or_create(conn, 'bpla_types', 'type_id', 'type_name', model.type_name)
                status_id = None
                if model.status_name:
                    status_id = self._get_or_create(conn, 'statuses', 'status_id', 'status_name', model.status_name)

                existing_id = self._model_exists(conn, model.model_name, manufacturer_id)
                if existing_id and not update_existing:
                    self._log(source_name, model, 'duplicate', f"Модель уже существует (ID={existing_id})")
                    return False, f"Дубликат: модель '{model.model_name}' уже в БД (ID={existing_id})"

                model_data = {
                    'model_name': model.model_name,
                    'manufacturer_id': manufacturer_id,
                    'type_id': type_id,
                    'status_id': status_id,
                    'payload_capacity_kg': model.payload_capacity_kg,
                    'takeoff_mass_kg': model.takeoff_mass_kg,
                    'flight_time_min': model.flight_time_min,
                    'max_speed_kmh': model.max_speed_kmh,
                    'max_range_km': model.max_range_km,
                    'max_altitude_m': model.max_altitude_m,
                    'wingspan_or_rotor_m': model.wingspan_or_rotor_m,
                    'length_m': model.length_m,
                    'battery_capacity_kwh': model.battery_capacity_kwh,
                    'fuel_consumption_l_per_h': model.fuel_consumption_l_per_h,
                    'wind_resistance_ms': model.wind_resistance_ms,
                    'labor_cost_per_ha': model.labor_cost_per_ha,
                    'energy_cost_per_ha_min': model.energy_cost_per_ha_min,
                    'energy_cost_per_ha_max': model.energy_cost_per_ha_max,
                    'recommended_field_area_min_ha': model.recommended_field_area_min_ha,
                    'recommended_field_area_max_ha': model.recommended_field_area_max_ha,
                    'ip_rating': model.ip_rating,
                    'temp_range_min_c': model.temp_range_min_c,
                    'temp_range_max_c': model.temp_range_max_c,
                    'power_type': model.power_type,
                    'source_url': model.source_url,
                    'notes': model.notes,
                }
                clean_data = {k: v for k, v in model_data.items() if v is not None}

                if existing_id and update_existing:
                    set_clause = ', '.join([f"{k} = ?" for k in clean_data.keys()])
                    values = list(clean_data.values()) + [existing_id]
                    conn.execute(f"UPDATE bpla_models SET {set_clause} WHERE model_id = ?", values)
                    model_id = existing_id
                    action = "обновлена"
                else:
                    cols = ', '.join(clean_data.keys())
                    placeholders = ', '.join(['?'] * len(clean_data))
                    cur = conn.execute(f"INSERT INTO bpla_models ({cols}) VALUES ({placeholders})",
                                       list(clean_data.values()))
                    model_id = cur.lastrowid
                    action = "создана"

                self._ingest_prices(conn, model_id, model.prices)
                conn.commit()
                self._log(source_name, model, 'success', f"Модель {action} (ID={model_id})")
                return True, f"Модель '{model.model_name}' успешно {action} (ID={model_id})"
        except Exception as e:
            logger.exception("Ошибка записи в БД")
            self._log(source_name, model, 'error', str(e))
            return False, f"Ошибка БД: {str(e)}"

    def _ingest_prices(self, conn, model_id, prices):
        for price in prices:
            supplier_name = price.get('supplier_name')
            supplier_id = None
            if supplier_name:
                supplier_id = self._get_or_create(conn, 'suppliers', 'supplier_id', 'supplier_name', supplier_name)
            conn.execute(
                """INSERT INTO prices (model_id, supplier_id, price_rub, price_type, currency,
                    configuration, availability, delivery_time_days, vat_included, warranty_months,
                    price_date, source_url, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (model_id, supplier_id, price.get('price_rub'), price.get('price_type', 'retail'),
                 price.get('currency', 'RUB'), price.get('configuration'), price.get('availability'),
                 price.get('delivery_time_days'), price.get('vat_included', 1),
                 price.get('warranty_months'), price.get('price_date', datetime.now().isoformat()),
                 price.get('source_url'), price.get('notes'))
            )

    def _log(self, source_name: str, model: UAVModelRaw, status: str, message: str):
        try:
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO collection_logs (source_name, url, model_name, manufacturer_name, status, message, raw_data_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (source_name, model.source_url, model.model_name, model.manufacturer_name,
                     status, message, json.dumps(asdict(model), ensure_ascii=False, default=str))
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Не удалось записать лог: {e}")

    def get_logs(self, limit: int = 100, status: Optional[str] = None) -> pd.DataFrame:
        with self._connect() as conn:
            sql = "SELECT * FROM collection_logs"
            params = []
            if status:
                sql += " WHERE status = ?"
                params.append(status)
            sql += " ORDER BY collected_at DESC LIMIT ?"
            params.append(limit)
            return pd.read_sql_query(sql, conn, params=params)

    def import_csv(self, csv_path: str, mapping: Dict[str, str],
                   source_name: str = "csv_import") -> Tuple[int, int, List[str]]:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        success = 0
        skipped = 0
        errors = []
        for idx, row in df.iterrows():
            try:
                model = UAVModelRaw(
                    model_name=str(row.get(mapping.get('model_name', 'model_name'), '')).strip(),
                    manufacturer_name=str(row.get(mapping.get('manufacturer_name', 'manufacturer_name'), '')).strip(),
                )
                if not model.model_name or not model.manufacturer_name:
                    skipped += 1
                    continue
                field_map = {
                    'country_name': 'country_name', 'type_name': 'type_name',
                    'payload_capacity_kg': 'payload_capacity_kg', 'takeoff_mass_kg': 'takeoff_mass_kg',
                    'flight_time_min': 'flight_time_min', 'max_speed_kmh': 'max_speed_kmh',
                    'max_range_km': 'max_range_km', 'max_altitude_m': 'max_altitude_m',
                    'wingspan_or_rotor_m': 'wingspan_or_rotor_m', 'length_m': 'length_m',
                    'battery_capacity_kwh': 'battery_capacity_kwh',
                    'fuel_consumption_l_per_h': 'fuel_consumption_l_per_h',
                    'wind_resistance_ms': 'wind_resistance_ms', 'labor_cost_per_ha': 'labor_cost_per_ha',
                    'energy_cost_per_ha_min': 'energy_cost_per_ha_min',
                    'energy_cost_per_ha_max': 'energy_cost_per_ha_max',
                    'recommended_field_area_min_ha': 'recommended_field_area_min_ha',
                    'recommended_field_area_max_ha': 'recommended_field_area_max_ha',
                    'ip_rating': 'ip_rating', 'temp_range_min_c': 'temp_range_min_c',
                    'temp_range_max_c': 'temp_range_max_c', 'power_type': 'power_type',
                    'source_url': 'source_url', 'notes': 'notes',
                }
                for csv_col, model_attr in field_map.items():
                    if csv_col in mapping:
                        val = row.get(mapping[csv_col])
                        if pd.notna(val):
                            setattr(model, model_attr, val)
                if 'price_rub' in mapping and pd.notna(row.get(mapping['price_rub'])):
                    model.prices.append({
                        'price_rub': DataValidator.clean_numeric(row.get(mapping['price_rub'])),
                        'supplier_name': row.get(mapping.get('supplier_name', 'supplier_name'), 'Импорт'),
                        'price_type': 'retail', 'currency': 'RUB'
                    })
                if 'applications' in mapping and pd.notna(row.get(mapping['applications'])):
                    apps = str(row.get(mapping['applications'])).split(';')
                    for a in apps:
                        model.applications.append({'application_name': a.strip(), 'application_category': 'Импорт'})
                ok, msg = self.ingest(model, source_name=source_name)
                if ok:
                    success += 1
                else:
                    if 'Дубликат' in msg:
                        skipped += 1
                    else:
                        errors.append(f"Строка {idx + 2}: {msg}")
            except Exception as e:
                errors.append(f"Строка {idx + 2}: {str(e)}")
        return success, skipped, errors


class WebDataCollector:
    def __init__(self, ingestor: DatabaseIngestor):
        self.ingestor = ingestor

    def collect_from_url(self, url: str, parser_type: str = 'generic_table',
                         mapping: Optional[Dict] = None,
                         headers: Optional[Dict] = None) -> Tuple[int, int, List[str]]:
        return 0, 0, ["Устаревший метод — используйте SmartScraper"]


def get_default_csv_mapping() -> Dict[str, str]:
    return {
        'model_name': 'model_name', 'manufacturer_name': 'manufacturer_name',
        'country_name': 'country_name', 'type_name': 'type_name',
        'payload_capacity_kg': 'payload_capacity_kg', 'takeoff_mass_kg': 'takeoff_mass_kg',
        'flight_time_min': 'flight_time_min', 'max_speed_kmh': 'max_speed_kmh',
        'max_range_km': 'max_range_km', 'max_altitude_m': 'max_altitude_m',
        'wingspan_or_rotor_m': 'wingspan_or_rotor_m', 'length_m': 'length_m',
        'battery_capacity_kwh': 'battery_capacity_kwh', 'fuel_consumption_l_per_h': 'fuel_consumption_l_per_h',
        'wind_resistance_ms': 'wind_resistance_ms', 'labor_cost_per_ha': 'labor_cost_per_ha',
        'energy_cost_per_ha_min': 'energy_cost_per_ha_min', 'energy_cost_per_ha_max': 'energy_cost_per_ha_max',
        'recommended_field_area_min_ha': 'recommended_field_area_min_ha',
        'recommended_field_area_max_ha': 'recommended_field_area_max_ha',
        'ip_rating': 'ip_rating', 'temp_range_min_c': 'temp_range_min_c',
        'temp_range_max_c': 'temp_range_max_c', 'power_type': 'power_type',
        'source_url': 'source_url', 'notes': 'notes',
        'price_rub': 'price_rub', 'supplier_name': 'supplier_name',
        'applications': 'applications',
    }


# ==================== УЛУЧШЕННЫЙ ИНТЕЛЛЕКТУАЛЬНЫЙ СКРАПЕР (v3) ====================

DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
}
REQUEST_DELAY = 1.5
TIMEOUT = 30
MAX_RETRIES = 3


@dataclass
class ScrapedUAV:
    model_name: Optional[str] = None
    manufacturer_name: Optional[str] = None
    source_url: str = ""
    source_site: str = ""
    scraped_at: str = field(default_factory=lambda: datetime.now().isoformat())
    payload_capacity_kg: Optional[float] = None
    takeoff_mass_kg: Optional[float] = None
    flight_time_min: Optional[float] = None
    max_speed_kmh: Optional[float] = None
    max_range_km: Optional[float] = None
    max_altitude_m: Optional[float] = None
    wingspan_or_rotor_m: Optional[float] = None
    length_m: Optional[float] = None
    battery_capacity_kwh: Optional[float] = None
    fuel_consumption_l_per_h: Optional[float] = None
    wind_resistance_ms: Optional[float] = None
    power_type: Optional[str] = None
    ip_rating: Optional[str] = None
    temp_range_min_c: Optional[float] = None
    temp_range_max_c: Optional[float] = None
    labor_cost_per_ha: Optional[float] = None
    energy_cost_per_ha_min: Optional[float] = None
    energy_cost_per_ha_max: Optional[float] = None
    recommended_field_area_min_ha: Optional[float] = None
    recommended_field_area_max_ha: Optional[float] = None
    price_rub: Optional[float] = None
    price_usd: Optional[float] = None
    price_currency: Optional[str] = None
    country_name: Optional[str] = None
    type_name: Optional[str] = None
    status_name: Optional[str] = "Активная"
    description: Optional[str] = None
    image_url: Optional[str] = None
    notes: Optional[str] = None
    raw_specs: Dict = field(default_factory=dict)

    def to_uav_model_raw(self):
        model = UAVModelRaw(
            model_name=self.model_name or "",
            manufacturer_name=self.manufacturer_name or "",
            country_name=self.country_name,
            type_name=self.type_name,
            status_name=self.status_name,
            payload_capacity_kg=self.payload_capacity_kg,
            takeoff_mass_kg=self.takeoff_mass_kg,
            flight_time_min=self.flight_time_min,
            max_speed_kmh=self.max_speed_kmh,
            max_range_km=self.max_range_km,
            max_altitude_m=self.max_altitude_m,
            wingspan_or_rotor_m=self.wingspan_or_rotor_m,
            length_m=self.length_m,
            battery_capacity_kwh=self.battery_capacity_kwh,
            fuel_consumption_l_per_h=self.fuel_consumption_l_per_h,
            wind_resistance_ms=self.wind_resistance_ms,
            power_type=self.power_type,
            ip_rating=self.ip_rating,
            temp_range_min_c=self.temp_range_min_c,
            temp_range_max_c=self.temp_range_max_c,
            labor_cost_per_ha=self.labor_cost_per_ha,
            energy_cost_per_ha_min=self.energy_cost_per_ha_min,
            energy_cost_per_ha_max=self.energy_cost_per_ha_max,
            recommended_field_area_min_ha=self.recommended_field_area_min_ha,
            recommended_field_area_max_ha=self.recommended_field_area_max_ha,
            source_url=self.source_url,
            notes=self.notes or self.description,
        )
        if self.price_rub or self.price_usd:
            model.prices.append({
                'price_rub': self.price_rub,
                'price_usd': self.price_usd,
                'currency': self.price_currency or 'RUB',
                'price_type': 'retail',
                'supplier_name': self.manufacturer_name or self.source_site,
                'source_url': self.source_url
            })
        return model


class SmartScraper:
    """
    Гибкий скрапер с многоуровневым анализом.
    Принцип: извлекать ТОЛЬКО то, что явно указано на странице.
    Никаких доменных эвристик, предположений о типе/стране/производителе.
    """

    # Расширенные паттерны для определения полей (русский + английский)
    FIELD_PATTERNS = {
        'model_name': [
            'model', 'name', 'модель', 'наименование', 'название', 'product', 'продукт',
            'drone', 'бпла', 'агродрон', 'uav', 'series', 'серия', 'item', 'артикул',
            'sku', 'product name', 'наименование изделия', 'variant', 'модификация'
        ],
        'manufacturer_name': [
            'manufacturer', 'brand', 'maker', 'производитель', 'бренд', 'компания', 'company',
            'vendor', 'изготовитель', 'factory', 'фирма', 'marque', 'trade mark', 'торговая марка',
            'developed by', 'разработчик'
        ],
        'payload_capacity_kg': [
            'payload', 'load', 'capacity', 'грузоподъемность', 'грузоподъёмность', 'нагрузка',
            'payload capacity', 'rated payload', 'max payload', 'spraying load', 'рабочая нагрузка',
            'spray load', 'tank capacity', 'объем бака', 'ёмкость бака', 'емкость бака',
            'liquid tank', 'бак', 'tank volume', 'load capacity', 'грузовместимость',
            ' spraying volume', 'объём распыла', 'рейтинговая нагрузка'
        ],
        'takeoff_mass_kg': [
            'takeoff', 'mtow', 'weight', 'mass', 'взлётная', 'взлетная', 'масса', 'вес',
            'max takeoff', 'all up weight', 'взлётная масса', 'взлетная масса', 'общий вес',
            'max weight', 'максимальная масса', 'total weight', 'gross weight', 'снаряженная масса',
            'max takeoff weight', 'макс. взлётная масса'
        ],
        'flight_time_min': [
            'flight time', 'endurance', 'duration', 'время полёта', 'время полета', 'полёт',
            'hover time', 'flight endurance', 'max flight time', 'автономность', 'длительность',
            'operating time', 'рабочее время', 'flight duration', 'полетное время', 'время работы',
            'время полета без подзарядки', 'max flight endurance'
        ],
        'max_speed_kmh': [
            'speed', 'velocity', 'скорость', 'max speed', 'cruise speed', 'макс. скорость',
            'максимальная скорость', 'скорость полёта', 'flight speed', 'airspeed', 'max velocity',
            'operating speed', 'рабочая скорость', 'скорость крейсерская'
        ],
        'max_range_km': [
            'range', 'дальность', 'control range', 'flight range', 'max range', 'дальность полёта',
            'transmission range', 'радиус действия', 'operating radius', 'дальность управления',
            'communication range', 'link range', 'дальность связи', 'max control range'
        ],
        'max_altitude_m': [
            'altitude', 'height', 'высота', 'max altitude', 'service ceiling', 'рабочая высота',
            'flight height', 'высота полёта', 'потолок', 'operating altitude', 'max height',
            'ceiling', 'практический потолок', 'relative height', 'относительная высота',
            'max flight altitude'
        ],
        'wingspan_or_rotor_m': [
            'wingspan', 'rotor', 'diameter', 'размах', 'диаметр', 'rotor diameter', 'propeller',
            'wheelbase', 'wheel base', 'база', 'диаметр ротора', 'размах крыла', 'prop diameter',
            'rotor span', 'размах винта', 'overall dimensions', 'габариты', 'dimensions',
            'diagonal wheelbase', 'диагональ'
        ],
        'length_m': [
            'length', 'длина', 'overall length', 'fuselage', 'фюзеляж', 'габаритная длина',
            'body length', 'длина корпуса', 'size', 'размер', 'overall size'
        ],
        'battery_capacity_kwh': [
            'battery', 'аккумулятор', 'capacity', 'ёмкость', 'емкость', 'battery capacity',
            'mah', 'wh', 'квт·ч', 'квтч', 'energy', 'энергия', 'battery energy',
            'power capacity', 'battery power', 'аккумуляторная', 'battery spec', 'battery type'
        ],
        'fuel_consumption_l_per_h': [
            'fuel', 'consumption', 'расход', 'топливо', 'fuel consumption', 'расход топлива',
            'fuel flow', 'fuel rate', 'fuel usage', 'fuel burn', 'удельный расход',
            'fuel consumption rate'
        ],
        'wind_resistance_ms': [
            'wind', 'ветер', 'resistance', 'wind resistance', 'ветроустойчивость',
            'max wind', 'wind speed', 'скорость ветра', 'порыв ветра', 'wind tolerance',
            'wind level', 'уровень ветра', 'wind rating', 'класс ветра', 'max wind speed',
            'wind resistance level'
        ],
        'power_type': [
            'power', 'engine', 'двигатель', 'тип двигателя', 'propulsion', 'силовая установка',
            'motor', 'electric', 'battery powered', 'fuel', 'гибрид', 'hybrid', 'бензин', 'дизель',
            'power system', 'силовая', 'power plant', 'тип питания', 'energy source',
            'powertrain', 'силовая установка'
        ],
        'ip_rating': [
            'ip', 'защита', 'protection', 'ingress protection', 'ip rating', 'ip code',
            'waterproof', 'dustproof', 'пылевлагозащита', 'защита от пыли', 'влагозащита',
            'protection class', 'класс защиты', 'environmental protection', 'степень защиты'
        ],
        'temp_range_min_c': [
            'temp min', 'min temp', 'темп мин', 'мин температура', 'min temperature',
            'operating temp', 'рабочая температура', 'temp range', 'temperature range',
            'min operating temp', 'минимальная температура', 'low temp', 'низкая температура',
            'working temperature min'
        ],
        'temp_range_max_c': [
            'temp max', 'max temp', 'темп макс', 'макс температура', 'max temperature',
            'operating temp', 'рабочая температура', 'max operating temp', 'максимальная температура',
            'high temp', 'высокая температура', 'working temperature max'
        ],
        'labor_cost_per_ha': [
            'labor', 'трудозатраты', 'man hour', 'чел.ч', 'person hour', 'трудоёмкость',
            'workload', 'operator', 'оператор', 'crew', 'экипаж', 'personnel', 'персонал',
            'labour cost', 'трудозатраты на га', 'man-hour per hectare'
        ],
        'energy_cost_per_ha_min': [
            'energy cost', 'энергозатраты', 'energy consumption', 'power consumption',
            'fuel consumption per ha', 'расход энергии', 'energy per hectare', 'энергия на га',
            'energy consumption min', 'min energy cost'
        ],
        'energy_cost_per_ha_max': [
            'energy cost max', 'энергозатраты макс', 'max energy consumption', 'max power consumption',
            'fuel consumption per ha max', 'расход энергии макс', 'max energy per hectare',
            'энергия на га макс', 'energy consumption max', 'max energy cost'
        ],
        'recommended_field_area_min_ha': [
            'field min', 'min area', 'площадь мин', 'мин площадь', 'recommended field',
            'field size min', 'минимальная площадь', 'min field', 'min area size',
            'operating area min', 'минимальная зона', 'min coverage area'
        ],
        'recommended_field_area_max_ha': [
            'field max', 'max area', 'площадь макс', 'макс площадь', 'recommended field',
            'field size max', 'максимальная площадь', 'max field', 'max area size',
            'operating area max', 'максимальная зона', 'max coverage', 'max coverage area'
        ],
        'price_rub': [
            'price', 'цена', 'cost', 'стоимость', 'retail price', 'dealer price',
            'msrp', 'руб', 'rub', 'usd', 'eur', '₽', '$', '€', 'price list', 'прайс',
            'purchase price', 'закупочная цена', 'selling price', 'продажная цена',
            'price rub', 'price usd', 'price eur', 'стоимость комплекта'
        ],
    }

    # Паттерны для определения, что таблица/блок содержит характеристики
    SPEC_INDICATORS = [
        'spec', 'param', 'feature', 'tech', 'charact', 'detail', 'property',
        'характеристика', 'параметр', 'спецификация', 'свойство', 'описание',
        'overview', 'summary', 'info', 'данные', 'data', 'metric', 'measurement',
        'specification', 'technical', 'технические'
    ]

    # Явно негативные паттерны (чтобы не принимать за характеристики)
    NEGATIVE_INDICATORS = [
        'review', 'отзыв', 'comment', 'комментарий', 'rating', 'рейтинг', 'vote', 'голосование',
        'related', 'похожие', 'recommended', 'рекомендуем', 'accessory', 'аксессуар',
        'news', 'новости', 'article', 'статья', 'blog', 'блог'
    ]

    def __init__(self, cache_dir: str = "./scraping_cache"):
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.last_request_time = 0
        self.debug_info: List[str] = []

    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self.last_request_time = time.time()

    def _get(self, url: str, retries: int = MAX_RETRIES) -> Optional[str]:
        cache_key = hashlib.md5(url.encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.html"
        if cache_file.exists():
            age = time.time() - cache_file.stat().st_mtime
            if age < 86400:
                self.debug_info.append(f"[CACHE] {url}")
                return cache_file.read_text(encoding='utf-8')
        for attempt in range(retries):
            try:
                self._rate_limit()
                resp = self.session.get(url, timeout=TIMEOUT)
                resp.raise_for_status()
                cache_file.write_text(resp.text, encoding='utf-8')
                self.debug_info.append(f"[GET] {url} — {len(resp.text)} chars")
                return resp.text
            except Exception as e:
                self.debug_info.append(f"[ERROR] {url} attempt {attempt + 1}: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None
        return None

    def _extract_number(self, text: str) -> Optional[float]:
        """Извлекает число, корректно обрабатывая разделители тысяч и десятичных."""
        if not text:
            return None
        text = str(text)
        # Удаляем неразрывные пробелы и пробелы между цифрами (разделители тысяч)
        text = re.sub(r'(\d)[\s\u202f\xa0](\d{3})', r'\1\2', text)
        # Ищем число (с поддержкой запятой как десятичного разделителя)
        match = re.search(r"[-+]?\d+(?:[.,]\d+)?", text.replace(',', '.'))
        if match:
            try:
                val = match.group().replace(',', '.')
                # Проверка: если это год или артикул (слишком большое целое без контекста) — пропускаем
                num = float(val)
                if num > 100000 and '.' not in val:
                    return None  # Скорее всего артикул или год
                return num
            except ValueError:
                return None
        return None

    def _extract_number_with_unit(self, text: str) -> Tuple[Optional[float], Optional[str]]:
        """Извлекает число и единицу измерения."""
        num = self._extract_number(text)
        if num is None:
            return None, None
        text_lower = str(text).lower()
        # Расширенный список единиц
        unit_patterns = [
            ('кг', 'kg'), ('kg', 'kg'), ('г ', 'g'), ('g ', 'g'), ('л', 'l'), ('l', 'l'),
            ('литр', 'l'), ('liter', 'l'), ('м ', 'm'), ('m ', 'm'), ('км', 'km'), ('km', 'km'),
            ('ч ', 'h'), ('h ', 'h'), ('мин', 'min'), ('min', 'min'), ('с ', 's'), ('s ', 's'),
            ('м/с', 'm/s'), ('мс', 'ms'), ('ms', 'ms'), ('°c', '°c'), ('°', '°'),
            ('руб', 'rub'), ('₽', 'rub'), ('rub', 'rub'), ('usd', 'usd'), ('$', 'usd'),
            ('eur', 'eur'), ('€', 'eur'), ('%', '%'), ('га', 'ha'), ('ha', 'ha'),
            ('квт·ч', 'kwh'), ('квтч', 'kwh'), ('kwh', 'kwh'), ('wh', 'wh'), ('mah', 'mah'),
            ('вт', 'w'), ('w ', 'w'), ('л/ч', 'l/h'), ('l/h', 'l/h'), ('м²', 'm2'), ('m2', 'm2'),
            ('мм', 'mm'), ('mm', 'mm'), ('см', 'cm'), ('cm', 'cm')
        ]
        for pattern, unit in unit_patterns:
            if pattern in text_lower:
                return num, unit
        return num, None

    def _extract_ip(self, text: str) -> Optional[str]:
        match = re.search(r'IP\s*(\d+)(?:\s*X)?', str(text), re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _match_field(self, key_text: str) -> Optional[str]:
        """Определяет поле по ключу с проверкой на целые слова/фразы."""
        if not key_text or len(key_text) > 100:
            return None
        key_lower = key_text.lower().strip().rstrip(':;.,-–—')
        for field_name, patterns in self.FIELD_PATTERNS.items():
            for pattern in patterns:
                # Проверяем точное вхождение фразы
                if pattern.lower() in key_lower:
                    # Дополнительная проверка: убедимся, что это не случайное вхождение
                    # (например, "speed" в "speedometer" — исключаем)
                    idx = key_lower.find(pattern.lower())
                    before = key_lower[idx - 1] if idx > 0 else ' '
                    after = key_lower[idx + len(pattern)] if idx + len(pattern) < len(key_lower) else ' '
                    if before in ' \t\n\r(-[' and after in ' \t\n\r:;.,-–—)]/':
                        return field_name
        return None

    def _is_specs_container(self, element) -> bool:
        """Проверяет, является ли элемент контейнером характеристик."""
        classes = element.get('class', [])
        if classes:
            class_str = ' '.join(classes).lower()
            # Проверяем негативные индикаторы
            for neg in self.NEGATIVE_INDICATORS:
                if neg in class_str:
                    return False
            for ind in self.SPEC_INDICATORS:
                if ind in class_str:
                    return True
        elem_id = element.get('id', '').lower()
        for neg in self.NEGATIVE_INDICATORS:
            if neg in elem_id:
                return False
        for ind in self.SPEC_INDICATORS:
            if ind in elem_id:
                return True
        for attr in element.attrs:
            if 'data' in attr.lower():
                val = str(element.get(attr, '')).lower()
                for neg in self.NEGATIVE_INDICATORS:
                    if neg in val:
                        return False
                for ind in self.SPEC_INDICATORS:
                    if ind in val:
                        return True
        return False

    def _parse_tables(self, soup: BeautifulSoup) -> List[Dict]:
        """Парсит все таблицы с улучшенной эвристикой (горизонтальные и вертикальные)."""
        results = []
        for table in soup.find_all('table'):
            rows = table.find_all('tr')
            if len(rows) < 1:
                continue

            # Собираем заголовки
            header_row = rows[0]
            headers = []
            for th in header_row.find_all(['th', 'td']):
                text = th.get_text(strip=True).lower()
                headers.append(text)

            # Проверяем, похожа ли таблица на характеристики
            is_specs = False
            for h in headers:
                if self._match_field(h):
                    is_specs = True
                    break

            # Если заголовки не определили, проверяем первые ячейки строк
            if not is_specs and len(rows) > 1:
                for row in rows[1:3]:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        first_cell = cells[0].get_text(strip=True).lower()
                        if self._match_field(first_cell):
                            is_specs = True
                            break

            if not is_specs:
                continue

            # Формат 1: ключ-значение (2 колонки, ключ в первой)
            if len(headers) <= 2 or all(h == '' for h in headers):
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        key = cells[0].get_text(strip=True)
                        val = cells[1].get_text(strip=True)
                        field = self._match_field(key)
                        if field:
                            results.append({field: val, 'model_name': None, 'manufacturer_name': None})
            else:
                # Формат 2: таблица с заголовками (много колонок)
                # Проверяем, есть ли колонки model_name/manufacturer
                has_model_col = any(self._match_field(h) == 'model_name' for h in headers)
                for row in rows[1:]:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) < 2:
                        continue
                    row_data = {}
                    for i, cell in enumerate(cells):
                        if i < len(headers):
                            field = self._match_field(headers[i])
                            if field:
                                row_data[field] = cell.get_text(strip=True)
                    if row_data:
                        results.append(row_data)

                # Формат 3: вертикальная таблица (первая колонка — названия параметров)
                if not has_model_col and len(rows) > 1 and len(headers) > 1:
                    # Первая строка — заголовки, первая колонка — параметры
                    # Но данные идут по строкам: Param | Value1 | Value2 ...
                    # Пока не поддерживаем множественные модели в одной вертикальной таблице
                    pass

        self.debug_info.append(f"[TABLES] Найдено {len(results)} записей")
        return results

    def _parse_dl_lists(self, soup: BeautifulSoup) -> List[Dict]:
        """Парсит списки определений (dl/dt/dd)."""
        results = []
        for dl in soup.find_all('dl'):
            dts = dl.find_all('dt')
            dds = dl.find_all('dd')
            if len(dts) == 0:
                continue
            row_data = {}
            for dt, dd in zip(dts, dds):
                key = dt.get_text(strip=True)
                field = self._match_field(key)
                if field:
                    row_data[field] = dd.get_text(strip=True)
            if row_data:
                results.append(row_data)
        self.debug_info.append(f"[DL] Найдено {len(results)} записей")
        return results

    def _parse_div_specs(self, soup: BeautifulSoup) -> List[Dict]:
        """Парсит div-блоки с характеристиками через эвристики label-value."""
        results = []

        # Стратегия 1: Контейнеры по классам/id
        for container in soup.find_all(['div', 'section', 'article']):
            if self._is_specs_container(container):
                items = container.find_all(['div', 'span', 'p', 'li'])
                row_data = {}
                for item in items:
                    text = item.get_text(strip=True)
                    if len(text) > 200:
                        continue
                    # Разделители: двоеточие, тире, em-dash
                    for sep in [':', '–', '-', '—']:
                        if sep in text:
                            parts = text.split(sep, 1)
                            if len(parts) == 2:
                                key, val = parts[0].strip(), parts[1].strip()
                                if 3 < len(key) < 60 and len(val) > 0:
                                    field = self._match_field(key)
                                    if field:
                                        row_data[field] = val
                                        break
                if row_data:
                    results.append(row_data)

        # Стратегия 2: Явные label-value пары по классам
        for row in soup.find_all(['div', 'span', 'li'], class_=re.compile(r'(row|item|line|entry|spec-row)', re.I)):
            label = row.find(['div', 'span', 'label', 'dt', 'th', 'b', 'strong', 'h4', 'h5'],
                             class_=re.compile(r'(label|name|key|title|param-name)', re.I))
            value = row.find(['div', 'span', 'dd', 'td', 'p', 'i', 'em', 'b', 'strong'],
                             class_=re.compile(r'(value|data|val|text|param-value)', re.I))
            if not label:
                # Fallback: ищем по структуре (первый текстовый элемент — label, второй — value)
                children = [c for c in row.children if c.name and c.get_text(strip=True)]
                if len(children) >= 2:
                    label, value = children[0], children[1]
            if label and value:
                key = label.get_text(strip=True)
                val = value.get_text(strip=True)
                field = self._match_field(key)
                if field:
                    results.append({field: val})

        # Стратегия 3: UL/LI с разделителями
        for li in soup.find_all('li'):
            text = li.get_text(strip=True)
            if len(text) > 150:
                continue
            for sep in [':', '–', '-', '—']:
                if sep in text and text.count(sep) == 1:
                    parts = text.split(sep, 1)
                    key, val = parts[0].strip(), parts[1].strip()
                    if 3 < len(key) < 60 and len(val) > 0:
                        field = self._match_field(key)
                        if field:
                            results.append({field: val})
                            break

        self.debug_info.append(f"[DIV] Найдено {len(results)} записей")
        return results

    def _parse_json_ld(self, soup: BeautifulSoup) -> List[Dict]:
        """Парсит JSON-LD (schema.org/Product, Vehicle, IndividualProduct)."""
        results = []
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                items = data if isinstance(data, list) else [data]
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    item_type = item.get('@type', '')
                    if isinstance(item_type, list):
                        item_type = item_type[0]
                    if item_type not in ['Product', 'Vehicle', 'IndividualProduct', 'Car']:
                        continue

                    row_data = {}
                    name = item.get('name', '')
                    brand = item.get('brand', {})
                    if isinstance(brand, dict):
                        brand = brand.get('name', '')
                    elif not isinstance(brand, str):
                        brand = ''

                    if name:
                        row_data['model_name'] = name
                    if brand:
                        row_data['manufacturer_name'] = brand

                    # Характеристики из additionalProperty / vehicleEngine / mileageFromOdometer
                    props = item.get('additionalProperty', [])
                    if isinstance(props, dict):
                        props = [props]
                    for prop in props:
                        if isinstance(prop, dict):
                            key = prop.get('name', '')
                            val = prop.get('value', '')
                            field = self._match_field(key)
                            if field:
                                row_data[field] = val

                    # Цена из offers
                    offers = item.get('offers', {})
                    if isinstance(offers, dict):
                        price = offers.get('price')
                        currency = offers.get('priceCurrency', '')
                        if price:
                            row_data['price_rub'] = price if currency in ['RUB', 'RUR', '₽'] else None
                            row_data['price_usd'] = price if currency in ['USD', '$'] else None
                            row_data['price_currency'] = currency

                    # Описание
                    desc = item.get('description', '')
                    if desc:
                        row_data['description'] = desc

                    if row_data:
                        results.append(row_data)
            except Exception:
                pass
        self.debug_info.append(f"[JSON-LD] Найдено {len(results)} записей")
        return results

    def _parse_meta_tags(self, soup: BeautifulSoup) -> Dict:
        """Извлекает данные из meta-тегов (Open Graph, Twitter, description)."""
        result = {}
        og_title = soup.find('meta', property='og:title')
        if og_title:
            content = og_title.get('content', '').strip()
            if content and '|' in content:
                parts = [p.strip() for p in content.split('|')]
                # Обычно: "Model Name | Brand" или "Brand Model Name"
                result['model_name'] = parts[0]
                if len(parts) > 1:
                    result['manufacturer_name'] = parts[-1]
            else:
                result['model_name'] = content

        # Twitter
        tw_title = soup.find('meta', attrs={'name': 'twitter:title'})
        if tw_title and not result.get('model_name'):
            result['model_name'] = tw_title.get('content', '').strip()

        # Description
        desc = soup.find('meta', attrs={'name': 'description'})
        if desc:
            result['description'] = desc.get('content', '').strip()

        # Keywords
        keywords = soup.find('meta', attrs={'name': 'keywords'})
        if keywords:
            result['keywords'] = keywords.get('content', '').strip()

        return result

    def _extract_model_name(self, soup: BeautifulSoup, url: str) -> Optional[str]:
        """Извлекает название модели ТОЛЬКО из явных источников на странице."""
        # 1. Из h1 (наиболее достоверный)
        h1 = soup.find('h1')
        if h1:
            text = h1.get_text(strip=True)
            if text and 3 < len(text) < 120:
                return text

        # 2. Из JSON-LD (Product.name)
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                if isinstance(data, dict) and data.get('name'):
                    name = data['name'].strip()
                    if name and 3 < len(name) < 120:
                        return name
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and item.get('name'):
                            name = item['name'].strip()
                            if name and 3 < len(name) < 120:
                                return name
            except Exception:
                pass

        # 3. Из title (только первая часть до разделителя)
        title = soup.find('title')
        if title:
            text = title.get_text(strip=True)
            for sep in [' | ', ' - ', ' – ', ' — ', ' :: ', ' // ', ' / ']:
                if sep in text:
                    text = text.split(sep)[0].strip()
                    break
            if text and 3 < len(text) < 120:
                return text

        # 4. Из Open Graph
        og = soup.find('meta', property='og:title')
        if og:
            text = og.get('content', '').strip()
            if text and 3 < len(text) < 120:
                return text

        # 5. Из первого заголовка h2 (если h1 отсутствует)
        h2 = soup.find('h2')
        if h2 and not h1:
            text = h2.get_text(strip=True)
            if text and 3 < len(text) < 120:
                return text

        return None

    def _extract_manufacturer(self, soup: BeautifulSoup, url: str) -> Optional[str]:
        """Извлекает производителя ТОЛЬКО из явных источников на странице."""
        # 1. Из JSON-LD brand
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                if isinstance(data, dict):
                    brand = data.get('brand', {})
                    if isinstance(brand, dict) and brand.get('name'):
                        return brand['name'].strip()
                    elif isinstance(brand, str):
                        return brand.strip()
                    # manufacturer внутри Product
                    mfg = data.get('manufacturer', {})
                    if isinstance(mfg, dict) and mfg.get('name'):
                        return mfg['name'].strip()
            except Exception:
                pass

        # 2. Из meta-тегов (author, twitter:creator, og:site_name)
        og_site = soup.find('meta', property='og:site_name')
        if og_site:
            text = og_site.get('content', '').strip()
            if text and 2 < len(text) < 60:
                return text

        author = soup.find('meta', attrs={'name': 'author'})
        if author:
            text = author.get('content', '').strip()
            if text and 2 < len(text) < 60:
                return text

        # 3. Из заголовка title (вторая часть после разделителя — часто бренд)
        title = soup.find('title')
        if title:
            text = title.get_text(strip=True)
            for sep in [' | ', ' - ', ' – ', ' — ', ' :: ']:
                if sep in text:
                    parts = [p.strip() for p in text.split(sep)]
                    if len(parts) >= 2:
                        # Последняя часть часто бренд, первая — модель
                        candidate = parts[-1]
                        if 2 < len(candidate) < 40 and not any(
                                x in candidate.lower() for x in ['купить', 'цена', 'отзыв', 'обзор']):
                            return candidate

        # 4. Из заголовков h1/h2 с явным указанием бренда
        for tag in ['h1', 'h2']:
            elem = soup.find(tag)
            if elem:
                text = elem.get_text(strip=True)
                # Если заголовок содержит "Brand Model" — попробуем выделить бренд
                # (первое слово часто бренд)
                words = text.split()
                if len(words) >= 2:
                    candidate = words[0]
                    if 2 < len(candidate) < 20 and candidate[0].isupper():
                        return candidate

        return None

    def _merge_results(self, results: List[Dict]) -> List[Dict]:
        """Объединяет результаты из разных источников по model_name."""
        merged = {}
        for row in results:
            model = row.get('model_name')
            if not model or model == 'Unknown':
                # Группируем безымянные записи отдельно по хешу
                model = f"__unnamed_{hash(tuple(row.items())) % 10000}"
            if model not in merged:
                merged[model] = {}
            for key, val in row.items():
                if val and (key not in merged[model] or not merged[model][key]):
                    merged[model][key] = val
        # Убираем безымянные ключи
        clean = []
        for k, v in merged.items():
            if not k.startswith('__unnamed_'):
                clean.append(v)
            else:
                # Для безымянных добавляем только если есть достаточно данных
                if len(v) >= 3:
                    clean.append(v)
        return clean

    def _normalize_value(self, field: str, raw_value: str) -> Any:
        """Нормализует значение с учётом поля и единиц измерения."""
        if not raw_value:
            return None

        if field == 'ip_rating':
            return self._extract_ip(raw_value)

        if field in ['model_name', 'manufacturer_name', 'type_name', 'power_type', 'status_name', 'description']:
            return raw_value.strip()

        # Числовые поля
        numeric_fields = [
            'payload_capacity_kg', 'takeoff_mass_kg', 'flight_time_min', 'max_speed_kmh',
            'max_range_km', 'max_altitude_m', 'wingspan_or_rotor_m', 'length_m',
            'battery_capacity_kwh', 'fuel_consumption_l_per_h', 'wind_resistance_ms',
            'labor_cost_per_ha', 'energy_cost_per_ha_min', 'energy_cost_per_ha_max',
            'recommended_field_area_min_ha', 'recommended_field_area_max_ha',
            'temp_range_min_c', 'temp_range_max_c', 'price_rub', 'price_usd'
        ]

        if field in numeric_fields:
            num, unit = self._extract_number_with_unit(raw_value)
            if num is None:
                return None

            # Конвертация единиц (осторожная)
            if unit == 'g' and field in ['payload_capacity_kg', 'takeoff_mass_kg']:
                return num / 1000
            if unit == 'cm' and field in ['wingspan_or_rotor_m', 'length_m']:
                return num / 100
            if unit == 'mm' and field in ['wingspan_or_rotor_m', 'length_m']:
                return num / 1000
            if unit == 'km' and field == 'max_range_km':
                return num  # Уже в нужных единицах
            if unit == 'm' and field == 'max_altitude_m':
                return num
            if unit == 'h' and field == 'flight_time_min':
                return num * 60
            if unit == 's' and field == 'flight_time_min':
                return num / 60

            # Для payload: если указаны литры (L) — это скорее всего объём бака
            # Не конвертируем в kg автоматически (чтобы избежать отсебятины),
            # но сохраняем число с пометкой в notes
            if unit in ['l', 'l/h'] and field == 'payload_capacity_kg':
                # Возвращаем число как есть, но позже можно добавить в notes
                return num

            return num

        return raw_value.strip()

    def scrape(self, url: str) -> List[ScrapedUAV]:
        """Главный метод сбора данных с URL."""
        self.debug_info = []
        self.debug_info.append(f"[START] Сбор с {url}")

        html = self._get(url)
        if not html:
            self.debug_info.append("[ERROR] Не удалось загрузить страницу")
            return []

        soup = BeautifulSoup(html, 'html.parser')

        # Собираем данные из всех источников
        all_data = []
        all_data.extend(self._parse_tables(soup))
        all_data.extend(self._parse_dl_lists(soup))
        all_data.extend(self._parse_div_specs(soup))
        all_data.extend(self._parse_json_ld(soup))

        # Добавляем meta-данные
        meta = self._parse_meta_tags(soup)
        if meta:
            all_data.append(meta)

        self.debug_info.append(f"[TOTAL] Сырых записей: {len(all_data)}")

        # Объединяем по моделям
        all_data = self._merge_results(all_data)
        self.debug_info.append(f"[MERGED] После объединения: {len(all_data)}")

        # Извлекаем fallback-значения из страницы (явные источники, не домен/URL)
        page_model_name = self._extract_model_name(soup, url)
        page_manufacturer = self._extract_manufacturer(soup, url)

        results = []
        for row_data in all_data:
            current_model = row_data.get('model_name') or page_model_name
            current_manufacturer = row_data.get('manufacturer_name') or page_manufacturer

            # СТРОГО: если нет явного названия модели или производителя — пропускаем
            if not current_model or not current_manufacturer:
                self.debug_info.append(f"[SKIP] Отсутствует model_name или manufacturer_name в записи")
                continue

            uav = ScrapedUAV(
                model_name=current_model,
                manufacturer_name=current_manufacturer,
                source_url=url,
                source_site=urlparse(url).netloc,
            )

            # Заполняем поля
            notes_parts = []
            for field, raw_value in row_data.items():
                if field in ('model_name', 'manufacturer_name'):
                    continue
                if hasattr(uav, field):
                    normalized = self._normalize_value(field, raw_value)
                    setattr(uav, field, normalized)

                    # Если payload был в литрах — добавим примечание
                    if field == 'payload_capacity_kg' and raw_value and any(
                            x in raw_value.lower() for x in ['л', 'l', 'литр', 'liter']):
                        notes_parts.append(f"Полезная нагрузка указана в литрах/объёме: {raw_value}")

            # Тип БПЛА: только если явно указан на странице. Никаких эвристик по wingspan/time!
            if not uav.type_name:
                # Проверим, не указан ли тип в тексте рядом с model_name
                # (например, "DJI Agras T40 — Мультироторный агродрон")
                pass  # Оставляем None, не догадываемся

            if notes_parts:
                uav.notes = "; ".join(notes_parts)

            results.append(uav)

        # Дедупликация по model_name + manufacturer_name
        seen = set()
        unique = []
        for r in results:
            key = f"{r.model_name}_{r.manufacturer_name}"
            if key not in seen:
                seen.add(key)
                unique.append(r)

        self.debug_info.append(f"[FINAL] Уникальных моделей: {len(unique)}")
        return unique


class ScrapingOrchestrator:
    def __init__(self, cache_dir: str = "./scraping_cache"):
        self.scraper = SmartScraper(cache_dir=cache_dir)
        self.results: List[ScrapedUAV] = []
        self.errors: List[str] = []
        self.debug_log: List[str] = []

    def scrape_url(self, url: str) -> Tuple[List[ScrapedUAV], List[str], List[str]]:
        self.results = []
        self.errors = []
        try:
            models = self.scraper.scrape(url)
            self.results = models
            self.debug_log = self.scraper.debug_info
            if not models:
                self.errors.append(f"Не удалось извлечь данные с {url}. Попробуйте страницу с таблицей характеристик.")
        except Exception as e:
            self.errors.append(f"{url}: {str(e)}")
        return self.results, self.errors, self.debug_log

    def scrape_all(self, urls: List[str], progress_callback=None) -> Tuple[List[ScrapedUAV], List[str], List[str]]:
        self.results = []
        self.errors = []
        all_debug = []
        for i, url in enumerate(urls):
            if progress_callback:
                progress_callback(i + 1, len(urls), url)
            models, errs, debug = self.scrape_url(url)
            self.results.extend(models)
            self.errors.extend(errs)
            all_debug.extend(debug)
        return self.results, self.errors, all_debug

    def save_to_database(self, db_path: str, update_existing: bool = False) -> Tuple[int, int, List[str]]:
        ingestor = DatabaseIngestor(db_path)
        success = 0
        skipped = 0
        errors = []
        for uav in self.results:
            try:
                model = uav.to_uav_model_raw()
                ok, msg = ingestor.ingest(model, source_name=uav.source_site, update_existing=update_existing)
                if ok:
                    success += 1
                else:
                    if 'Дубликат' in msg:
                        skipped += 1
                    else:
                        errors.append(f"{uav.model_name}: {msg}")
            except Exception as e:
                errors.append(f"{uav.model_name}: {str(e)}")
        return success, skipped, errors


# ==================== ПОИСК БАЗЫ ДАННЫХ ====================
def find_database():
    candidates = [
        Path(__file__).parent / "bpla_database.db",
        Path(__file__).parent / "БПЛА_сельхоз_реляционная_БД.db",
        Path.cwd() / "bpla_database.db",
        Path.cwd() / "БПЛА_сельхоз_реляционная_БД.db",
        Path.cwd().parent / "bpla_database.db",
        Path("bpla_database.db"),
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 10000:
            return p
    return None


db_file = find_database()

# ==================== ИНИЦИАЛИЗАЦИЯ БД ПРИ ОТСУТСТВИИ ====================
if db_file is None:
    st.error("❌ База данных НЕ НАЙДЕНА!")
    st.info("Ожидаемый файл: **bpla_database.db** или **БПЛА_сельхоз_реляционная_БД.db**")
    st.write("Текущая папка:", str(Path.cwd().absolute()))
    st.write("Файлы в текущей папке:", [f.name for f in Path.cwd().iterdir() if f.is_file()])

    st.divider()
    st.subheader("🆕 Создание новой базы данных")
    st.caption(
        "База данных отсутствует. Вы можете создать её автоматически со всей структурой и начальными справочными данными.")

    col1, col2 = st.columns(2)
    with col1:
        db_name = st.text_input("Имя файла БД", value="bpla_database.db")
    with col2:
        st.markdown("&nbsp;")
        create_btn = st.button("🔨 Создать базу данных", type="primary", width='stretch')

    if create_btn:
        init_sql = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS countries (country_id INTEGER PRIMARY KEY AUTOINCREMENT, country_code TEXT, country_name TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS manufacturers (manufacturer_id INTEGER PRIMARY KEY AUTOINCREMENT, country_id INTEGER, manufacturer_name TEXT NOT NULL, website TEXT, FOREIGN KEY (country_id) REFERENCES countries(country_id));
CREATE TABLE IF NOT EXISTS bpla_types (type_id INTEGER PRIMARY KEY AUTOINCREMENT, type_name TEXT NOT NULL, type_description TEXT, vtol_capability INTEGER DEFAULT 0, fixed_wing INTEGER DEFAULT 0, rotary_wing INTEGER DEFAULT 0, multirotor INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS statuses (status_id INTEGER PRIMARY KEY AUTOINCREMENT, status_name TEXT NOT NULL, status_description TEXT);
CREATE TABLE IF NOT EXISTS applications (application_id INTEGER PRIMARY KEY AUTOINCREMENT, application_category TEXT, application_name TEXT NOT NULL, description TEXT);
CREATE TABLE IF NOT EXISTS agro_zones (zone_id INTEGER PRIMARY KEY AUTOINCREMENT, zone_name TEXT NOT NULL, zone_category TEXT, recommended_field_area_max_ha REAL, recommended_field_area_min_ha REAL, terrain_complexity TEXT);
CREATE TABLE IF NOT EXISTS operations (operation_id INTEGER PRIMARY KEY AUTOINCREMENT, operation_name TEXT NOT NULL, operation_category TEXT, description TEXT, required_accuracy_m REAL, optimal_flight_height_m REAL, optimal_speed_ms REAL, season_timing TEXT);
CREATE TABLE IF NOT EXISTS sensors (sensor_id INTEGER PRIMARY KEY AUTOINCREMENT, sensor_name TEXT NOT NULL, sensor_type TEXT, resolution_mp REAL, spectral_bands TEXT, weight_kg REAL, power_consumption_w REAL, gimbal_stabilization INTEGER, real_time_transmission INTEGER);
CREATE TABLE IF NOT EXISTS suppliers (supplier_id INTEGER PRIMARY KEY AUTOINCREMENT, supplier_name TEXT NOT NULL, supplier_type TEXT, city TEXT, phone TEXT, email TEXT, website TEXT, inn TEXT, is_official INTEGER DEFAULT 0);

CREATE TABLE IF NOT EXISTS bpla_models (
    model_id INTEGER PRIMARY KEY AUTOINCREMENT, model_name TEXT NOT NULL, manufacturer_id INTEGER, type_id INTEGER, status_id INTEGER,
    payload_capacity_kg REAL, takeoff_mass_kg REAL, flight_time_min REAL, max_speed_kmh REAL, max_range_km REAL, max_altitude_m REAL,
    wingspan_or_rotor_m REAL, length_m REAL, battery_capacity_kwh REAL, fuel_consumption_l_per_h REAL, wind_resistance_ms REAL,
    power_type TEXT, ip_rating TEXT, temp_range_min_c REAL, temp_range_max_c REAL,
    labor_cost_per_ha REAL, energy_cost_per_ha_min REAL, energy_cost_per_ha_max REAL,
    recommended_field_area_min_ha REAL, recommended_field_area_max_ha REAL,
    source_url TEXT, notes TEXT,
    FOREIGN KEY (manufacturer_id) REFERENCES manufacturers(manufacturer_id),
    FOREIGN KEY (type_id) REFERENCES bpla_types(type_id),
    FOREIGN KEY (status_id) REFERENCES statuses(status_id)
);

CREATE TABLE IF NOT EXISTS model_applications (model_id INTEGER, application_id INTEGER, efficiency_score INTEGER, is_primary INTEGER DEFAULT 0, PRIMARY KEY (model_id, application_id), FOREIGN KEY (model_id) REFERENCES bpla_models(model_id) ON DELETE CASCADE, FOREIGN KEY (application_id) REFERENCES applications(application_id));
CREATE TABLE IF NOT EXISTS model_agro_zones (model_id INTEGER, zone_id INTEGER, suitability_score INTEGER, notes TEXT, PRIMARY KEY (model_id, zone_id), FOREIGN KEY (model_id) REFERENCES bpla_models(model_id) ON DELETE CASCADE, FOREIGN KEY (zone_id) REFERENCES agro_zones(zone_id));
CREATE TABLE IF NOT EXISTS model_operations (model_id INTEGER, operation_id INTEGER, is_suitable INTEGER DEFAULT 1, productivity_ha_per_hour REAL, spray_width_m REAL, liquid_capacity_l REAL, droplet_size_um REAL, application_rate_l_per_ha REAL, positioning_accuracy_cm REAL, overlap_percent REAL, skip_percent REAL, coverage_uniformity_cv REAL, suitability_score INTEGER, notes TEXT, PRIMARY KEY (model_id, operation_id), FOREIGN KEY (model_id) REFERENCES bpla_models(model_id) ON DELETE CASCADE, FOREIGN KEY (operation_id) REFERENCES operations(operation_id));
CREATE TABLE IF NOT EXISTS model_sensors (model_id INTEGER, sensor_id INTEGER, is_compatible INTEGER DEFAULT 1, is_default INTEGER DEFAULT 0, max_flight_time_with_sensor_min REAL, notes TEXT, PRIMARY KEY (model_id, sensor_id), FOREIGN KEY (model_id) REFERENCES bpla_models(model_id) ON DELETE CASCADE, FOREIGN KEY (sensor_id) REFERENCES sensors(sensor_id));
CREATE TABLE IF NOT EXISTS prices (price_id INTEGER PRIMARY KEY AUTOINCREMENT, model_id INTEGER NOT NULL, supplier_id INTEGER, price_rub REAL, price_type TEXT, currency TEXT DEFAULT 'RUB', configuration TEXT, availability TEXT, delivery_time_days INTEGER, vat_included INTEGER DEFAULT 1, warranty_months INTEGER, price_date TEXT, source_url TEXT, notes TEXT, FOREIGN KEY (model_id) REFERENCES bpla_models(model_id) ON DELETE CASCADE, FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id));
CREATE TABLE IF NOT EXISTS tco (tco_id INTEGER PRIMARY KEY AUTOINCREMENT, model_id INTEGER NOT NULL, ownership_period_years INTEGER DEFAULT 5, initial_cost_rub REAL, annual_maintenance_rub REAL, battery_replacement_cost_rub REAL, fuel_cost_per_hour_rub REAL, operator_cost_per_hour_rub REAL, insurance_cost_annual_rub REAL, certification_cost_rub REAL, total_tco_rub REAL, cost_per_ha_rub REAL, notes TEXT, FOREIGN KEY (model_id) REFERENCES bpla_models(model_id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS type_requirements (requirement_id INTEGER PRIMARY KEY AUTOINCREMENT, type_id INTEGER, parameter_name TEXT NOT NULL, parameter_value TEXT, parameter_unit TEXT, perspective_year INTEGER, requirement_category TEXT, FOREIGN KEY (type_id) REFERENCES bpla_types(type_id));
CREATE TABLE IF NOT EXISTS comparison_matrices (matrix_id INTEGER PRIMARY KEY AUTOINCREMENT, matrix_name TEXT, type_id_1 INTEGER, type_id_2 INTEGER, winner_type_id INTEGER, comparison_criteria TEXT, notes TEXT, FOREIGN KEY (type_id_1) REFERENCES bpla_types(type_id), FOREIGN KEY (type_id_2) REFERENCES bpla_types(type_id), FOREIGN KEY (winner_type_id) REFERENCES bpla_types(type_id));
CREATE TABLE IF NOT EXISTS agro_requirements (requirement_id INTEGER PRIMARY KEY AUTOINCREMENT, operation_id INTEGER, crop_type TEXT, growth_stage TEXT, max_flight_height_m REAL, min_flight_height_m REAL, max_wind_speed_ms REAL, max_temperature_c REAL, min_temperature_c REAL, buffer_zone_m REAL, required_overlap_percent REAL, notes TEXT, FOREIGN KEY (operation_id) REFERENCES operations(operation_id));
CREATE TABLE IF NOT EXISTS collection_logs (log_id INTEGER PRIMARY KEY AUTOINCREMENT, source_name TEXT, url TEXT, model_name TEXT, manufacturer_name TEXT, status TEXT, message TEXT, raw_data_json TEXT, collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);

INSERT OR IGNORE INTO bpla_types (type_id, type_name, type_description, vtol_capability, fixed_wing, rotary_wing, multirotor) VALUES
(1, 'Мультиротор', 'БПЛА с несколькими роторами', 0, 0, 0, 1),
(2, 'Самолётный', 'Фиксированное крыло', 0, 1, 0, 0),
(3, 'Вертолётный', 'Один или два несущих винта', 0, 0, 1, 0),
(4, 'Гибридный (VTOL)', 'Вертикальный взлёт + горизонтальный полёт', 1, 1, 0, 1);

INSERT OR IGNORE INTO statuses (status_id, status_name, status_description) VALUES
(1, 'Активная', 'Модель производится и продается'),
(2, 'Снята с производства', 'Модель больше не выпускается'),
(3, 'Прототип', 'Экспериментальная модель'),
(4, 'Ожидается', 'Анонсирована, но не выпущена');

INSERT OR IGNORE INTO applications (application_id, application_category, application_name, description) VALUES
(1, 'Опрыскивание', 'Химическая обработка', 'Нанесение пестицидов, гербицидов'),
(2, 'Опрыскивание', 'Биологическая обработка', 'Нанесение биопрепаратов'),
(3, 'Внесение удобрений', 'Разбрасывание удобрений', 'Сухое и жидкое внесение'),
(4, 'Внесение удобрений', 'Точечное внесение', 'Локализованное внесение'),
(5, 'Мониторинг', 'Мультиспектральная съёмка', 'NDVI, NDRE анализ'),
(6, 'Мониторинг', 'Тепловизионная съёмка', 'Инфракрасная диагностика'),
(7, 'Мониторинг', 'Фотографическая съёмка', 'RGB-съёмка'),
(8, 'Посев', 'Аэровысевающие работы', 'Высевающие операции'),
(9, 'Картографирование', '3D-моделирование', 'Цифровые модели рельефа'),
(10, 'Защита растений', 'Обработка семян', 'Предпосевная обработка');

INSERT OR IGNORE INTO operations (operation_id, operation_category, operation_name, description, required_accuracy_m, optimal_flight_height_m, optimal_speed_ms) VALUES
(1, 'Химическая защита', 'Опрыскивание', 'Нанесение ЗРС', 0.5, 3.0, 5.0),
(2, 'Химическая защита', 'Деструкция сорняков', 'Точечное нанесение', 0.1, 2.0, 3.0),
(3, 'Минеральное питание', 'Внесение жидких удобрений', 'Корневые подкормки', 1.0, 4.0, 6.0),
(4, 'Минеральное питание', 'Внесение гранулированных', 'Разбрасывание', 2.0, 5.0, 8.0),
(5, 'Мониторинг', 'Мультиспектральная съёмка', 'NDVI анализ', 0.05, 100.0, 10.0),
(6, 'Мониторинг', 'Тепловизионная съёмка', 'ИК диагностика', 0.1, 80.0, 8.0),
(7, 'Посев', 'Аэровысевающие работы', 'Высевающие операции', 0.5, 3.0, 4.0),
(8, 'Картографирование', 'Фотопланирование', 'Ортофотопланы', 0.02, 120.0, 12.0);

INSERT OR IGNORE INTO agro_zones (zone_id, zone_name, zone_category, recommended_field_area_min_ha, recommended_field_area_max_ha, terrain_complexity) VALUES
(1, 'Ровные поля', 'Рельеф', 0, 10000, 'низкая'),
(2, 'Холмистая местность', 'Рельеф', 0, 1000, 'средняя'),
(3, 'Горные склоны', 'Рельеф', 0, 500, 'высокая'),
(4, 'Малые поля', 'Размер', 0, 50, 'низкая'),
(5, 'Средние поля', 'Размер', 50, 500, 'низкая'),
(6, 'Крупные поля', 'Размер', 500, 5000, 'низкая'),
(7, 'Тепличные комплексы', 'Инфраструктура', 0, 100, 'низкая'),
(8, 'Виноградники', 'Культура', 0, 500, 'средняя'),
(9, 'Сады', 'Культура', 0, 300, 'средняя'),
(10, 'Зерновые культуры', 'Культура', 50, 5000, 'низкая');

INSERT OR IGNORE INTO sensors (sensor_id, sensor_name, sensor_type, resolution_mp, spectral_bands, weight_kg, power_consumption_w, gimbal_stabilization, real_time_transmission) VALUES
(1, 'RGB-камера 20 Мп', 'Оптическая', 20.0, 'RGB', 0.3, 5, 1, 1),
(2, 'Мультиспектральная M3M', 'Мультиспектральная', 5.0, 'RGB+NIR+RE', 0.5, 8, 1, 1),
(3, 'Тепловизор XT2', 'Тепловизионная', 0.3, 'LWIR', 0.4, 10, 1, 1),
(4, 'LiDAR Zenmuse L1', 'LiDAR', 0.0, 'NIR', 0.9, 25, 1, 0),
(5, 'Гиперспектральная Pika L', 'Гиперспектральная', 1.0, '400-1000nm', 1.2, 15, 0, 0);

CREATE INDEX IF NOT EXISTS idx_models_manufacturer ON bpla_models(manufacturer_id);
CREATE INDEX IF NOT EXISTS idx_models_type ON bpla_models(type_id);
CREATE INDEX IF NOT EXISTS idx_models_status ON bpla_models(status_id);
CREATE INDEX IF NOT EXISTS idx_prices_model ON prices(model_id);
CREATE INDEX IF NOT EXISTS idx_tco_model ON tco(model_id);
CREATE INDEX IF NOT EXISTS idx_logs_status ON collection_logs(status);
CREATE INDEX IF NOT EXISTS idx_logs_date ON collection_logs(collected_at);
"""

        try:
            new_db_path = Path.cwd() / db_name
            conn = sqlite3.connect(str(new_db_path))
            conn.executescript(init_sql)
            conn.close()

            st.success(f"✅ База данных успешно создана: `{new_db_path}`")
            st.info("Перезагрузите страницу для работы с новой базой данных.")

            conn = sqlite3.connect(str(new_db_path))
            tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn)
            views = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='view' ORDER BY name", conn)
            conn.close()

            col_t, col_v = st.columns(2)
            with col_t:
                st.markdown(f"**Таблиц создано: {len(tables)}**")
                st.dataframe(tables, hide_index=True, width='stretch')
            with col_v:
                st.markdown(f"**Представлений создано: {len(views)}**")
                st.dataframe(views, hide_index=True, width='stretch')

            st.balloons()

            if st.button("🔄 Перезагрузить приложение"):
                st.rerun()

        except Exception as e:
            st.error(f"❌ Ошибка создания БД: {e}")

    st.stop()

# ==================== ПОВТОРНЫЙ ПОИСК БД (если создана) ====================
db_file = find_database()

# ==================== ИНИЦИАЛИЗАЦИЯ ИНЖЕСТОРА ====================
if db_file:
    try:
        ingestor = DatabaseIngestor(str(db_file))
        web_collector = WebDataCollector(ingestor)
    except Exception as e:
        st.error(f"Ошибка инициализации модуля сбора: {e}")
        ingestor = None
        web_collector = None
else:
    ingestor = None
    web_collector = None

st.set_page_config(
    page_title="База данных БПЛА для АПК — Полная версия",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================== ПОИСК БАЗЫ ДАННЫХ ====================
def find_database():
    candidates = [
        Path(__file__).parent / "bpla_database.db",
        Path(__file__).parent / "БПЛА_сельхоз_реляционная_БД.db",
        Path.cwd() / "bpla_database.db",
        Path.cwd() / "БПЛА_сельхоз_реляционная_БД.db",
        Path.cwd().parent / "bpla_database.db",
        Path("bpla_database.db"),
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 10000:
            return p
    return None


db_file = find_database()

if db_file is None:
    st.error("❌ База данных НЕ НАЙДЕНА!")
    st.info("Ожидаемый файл: **bpla_database.db** или **БПЛА_сельхоз_реляционная_БД.db**")
    st.write("Текущая папка:", str(Path.cwd().absolute()))
    st.write("Файлы в текущей папке:", [f.name for f in Path.cwd().iterdir() if f.is_file()])
    st.stop()


# ==================== УТИЛИТЫ БД ====================
def get_table_columns(conn, table_name):
    try:
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        return [row[1] for row in cursor.fetchall()]
    except Exception:
        return []


def table_exists(conn, name):
    cur = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None


# ==================== ЗАГРУЗКА ДАННЫХ ====================
@st.cache_data(ttl=3600)
def load_data(db_path_str):
    conn = sqlite3.connect(db_path_str)
    existing = {t[0] for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

    # --- Основной датафрейм ---
    selects = ["m.*"]
    joins = []
    if "manufacturers" in existing:
        selects.append("man.manufacturer_name")
        joins.append("LEFT JOIN manufacturers man ON m.manufacturer_id = man.manufacturer_id")
    if "countries" in existing and "manufacturers" in existing:
        selects.append("c.country_name")
        joins.append("LEFT JOIN countries c ON man.country_id = c.country_id")
    if "bpla_types" in existing:
        selects += ["t.type_name", "t.type_description", "t.vtol_capability", "t.fixed_wing", "t.rotary_wing",
                    "t.multirotor"]
        joins.append("LEFT JOIN bpla_types t ON m.type_id = t.type_id")
    if "statuses" in existing:
        selects += ["s.status_name", "s.status_description"]
        joins.append("LEFT JOIN statuses s ON m.status_id = s.status_id")
    query = f"SELECT {', '.join(selects)} FROM bpla_models m {' '.join(joins)}"
    df = pd.read_sql_query(query, conn)

    # --- Цены ---
    df_prices = pd.DataFrame()
    if 'prices' in existing:
        sup_cols = get_table_columns(conn, "suppliers")
        sel = ["p.*"]
        j = ""
        if 'suppliers' in existing and 'supplier_name' in sup_cols:
            sel.append("s.supplier_name")
            j = "LEFT JOIN suppliers s ON p.supplier_id = s.supplier_id"
        try:
            df_prices = pd.read_sql_query(f"SELECT {', '.join(sel)} FROM prices p {j} WHERE p.price_rub IS NOT NULL",
                                          conn)
            if not df_prices.empty and 'model_id' in df_prices.columns and 'price_rub' in df_prices.columns:
                price_agg = df_prices.groupby('model_id').agg(
                    min_price=('price_rub', 'min'), max_price=('price_rub', 'max'),
                    avg_price=('price_rub', 'mean'), price_count=('price_rub', 'count')
                ).reset_index()
                df = df.merge(price_agg, on='model_id', how='left')
        except Exception:
            df_prices = pd.DataFrame()

    # --- Применения ---
    df_apps = pd.DataFrame()
    if 'model_applications' in existing and 'applications' in existing:
        try:
            df_apps = pd.read_sql_query("""
                SELECT ma.model_id, a.application_name, a.application_category, a.description,
                       ma.is_primary, ma.efficiency_score
                FROM model_applications ma JOIN applications a ON ma.application_id = a.application_id
            """, conn)
            apps_agg = pd.read_sql_query("""
                SELECT ma.model_id, GROUP_CONCAT(a.application_name) as applications,
                       GROUP_CONCAT(DISTINCT a.application_category) as app_categories,
                       GROUP_CONCAT(CASE WHEN ma.is_primary=1 THEN a.application_name END) as primary_apps
                FROM model_applications ma JOIN applications a ON ma.application_id = a.application_id
                GROUP BY ma.model_id
            """, conn)
            df = df.merge(apps_agg, on='model_id', how='left')
        except Exception:
            df_apps = pd.DataFrame()

    # --- Агро-зоны ---
    df_zones = pd.DataFrame()
    if 'model_agro_zones' in existing and 'agro_zones' in existing:
        try:
            df_zones = pd.read_sql_query("""
                SELECT maz.model_id, az.zone_name, az.zone_category, az.recommended_field_area_min_ha,
                       az.recommended_field_area_max_ha, az.terrain_complexity,
                       maz.suitability_score, maz.notes
                FROM model_agro_zones maz JOIN agro_zones az ON maz.zone_id = az.zone_id
            """, conn)
            zones_agg = pd.read_sql_query("""
                SELECT maz.model_id, GROUP_CONCAT(az.zone_name) as agro_zones,
                       GROUP_CONCAT(DISTINCT az.zone_category) as zone_categories,
                       MAX(maz.suitability_score) as max_suitability
                FROM model_agro_zones maz JOIN agro_zones az ON maz.zone_id = az.zone_id
                GROUP BY maz.model_id
            """, conn)
            df = df.merge(zones_agg, on='model_id', how='left')
        except Exception:
            df_zones = pd.DataFrame()

    # --- Операции ---
    df_ops = pd.DataFrame()
    if 'model_operations' in existing and 'operations' in existing:
        try:
            df_ops = pd.read_sql_query("""
                SELECT mo.*, o.operation_name, o.operation_category, o.description as op_description,
                       o.required_accuracy_m, o.optimal_flight_height_m, o.optimal_speed_ms, o.season_timing
                FROM model_operations mo JOIN operations o ON mo.operation_id = o.operation_id
                WHERE mo.is_suitable = 1
            """, conn)
            ops_agg = pd.read_sql_query("""
                SELECT mo.model_id, GROUP_CONCAT(o.operation_name) as operations,
                       GROUP_CONCAT(DISTINCT o.operation_category) as op_categories,
                       MAX(mo.suitability_score) as max_op_suitability,
                       MAX(mo.productivity_ha_per_hour) as max_productivity
                FROM model_operations mo JOIN operations o ON mo.operation_id = o.operation_id
                WHERE mo.is_suitable = 1 GROUP BY mo.model_id
            """, conn)
            df = df.merge(ops_agg, on='model_id', how='left')
        except Exception:
            df_ops = pd.DataFrame()

    # --- Сенсоры ---
    df_sensors = pd.DataFrame()
    if 'model_sensors' in existing and 'sensors' in existing:
        se_cols = get_table_columns(conn, "sensors")
        sensor_fields = ['sensor_name', 'sensor_type', 'resolution_mp', 'spectral_bands', 'weight_kg',
                         'power_consumption_w', 'gimbal_stabilization']
        avail_sensor_fields = [f for f in sensor_fields if f in se_cols]
        cols_sql = ", ".join([f"se.{f}" for f in avail_sensor_fields]) if avail_sensor_fields else ""
        ms_cols = get_table_columns(conn, "model_sensors")
        ms_fields = ['model_id', 'sensor_id', 'is_compatible', 'is_default', 'max_flight_time_with_sensor_min', 'notes']
        avail_ms_fields = [f for f in ms_fields if f in ms_cols]
        ms_sql = ", ".join([f"ms.{f}" for f in avail_ms_fields])
        try:
            q = f"SELECT {ms_sql}{', ' + cols_sql if cols_sql else ''} FROM model_sensors ms JOIN sensors se ON ms.sensor_id = se.sensor_id WHERE ms.is_compatible = 1"
            df_sensors = pd.read_sql_query(q, conn)
        except Exception:
            try:
                df_sensors = pd.read_sql_query("SELECT * FROM model_sensors WHERE is_compatible = 1", conn)
            except Exception:
                df_sensors = pd.DataFrame()

    # --- Агротребования ---
    df_reqs = pd.DataFrame()
    if 'agro_requirements' in existing and 'operations' in existing:
        try:
            df_reqs = pd.read_sql_query("""
                SELECT ar.*, o.operation_name as req_operation_name
                FROM agro_requirements ar LEFT JOIN operations o ON ar.operation_id = o.operation_id
            """, conn)
        except Exception:
            df_reqs = pd.DataFrame()

    # --- TCO ---
    df_tco = pd.DataFrame()
    if 'tco' in existing:
        try:
            df_tco = pd.read_sql_query("SELECT * FROM tco", conn)
        except Exception:
            df_tco = pd.DataFrame()

    # --- Требования к типам ---
    df_type_reqs = pd.DataFrame()
    if 'type_requirements' in existing and 'bpla_types' in existing:
        try:
            df_type_reqs = pd.read_sql_query("""
                SELECT tr.*, bt.type_name, bt.type_description
                FROM type_requirements tr
                LEFT JOIN bpla_types bt ON tr.type_id = bt.type_id
                ORDER BY bt.type_name, tr.requirement_category, tr.parameter_name
            """, conn)
        except Exception:
            df_type_reqs = pd.DataFrame()

    # --- Сравнительные матрицы ---
    df_comparison = pd.DataFrame()
    if 'comparison_matrices' in existing:
        try:
            df_comparison = pd.read_sql_query("""
                SELECT cm.*, t1.type_name as type_1_name, t2.type_name as type_2_name, tw.type_name as winner_name
                FROM comparison_matrices cm
                LEFT JOIN bpla_types t1 ON cm.type_id_1 = t1.type_id
                LEFT JOIN bpla_types t2 ON cm.type_id_2 = t2.type_id
                LEFT JOIN bpla_types tw ON cm.winner_type_id = tw.type_id
            """, conn)
        except Exception:
            df_comparison = pd.DataFrame()

    conn.close()

    for col in ['applications', 'agro_zones', 'operations', 'country_name', 'status_name',
                'app_categories', 'zone_categories', 'op_categories', 'primary_apps']:
        if col in df.columns:
            df[col] = df[col].fillna('—')

    return df, df_prices, df_apps, df_zones, df_ops, df_sensors, df_reqs, df_tco, df_type_reqs, df_comparison


@st.cache_data(ttl=3600)
def get_filter_options(db_path_str):
    conn = sqlite3.connect(db_path_str)
    existing = {t[0] for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

    def safe_query(q, fallback=None):
        try:
            return pd.read_sql_query(q, conn)
        except Exception:
            return fallback if fallback is not None else pd.DataFrame()

    types = safe_query("SELECT * FROM bpla_types ORDER BY type_name") if 'bpla_types' in existing else pd.DataFrame()
    countries = safe_query(
        "SELECT country_id, country_name FROM countries ORDER BY country_name") if 'countries' in existing else pd.DataFrame()
    manufacturers = safe_query(
        "SELECT manufacturer_id, manufacturer_name, country_id FROM manufacturers ORDER BY manufacturer_name") if 'manufacturers' in existing else pd.DataFrame()
    applications = safe_query(
        "SELECT application_id, application_name, application_category FROM applications ORDER BY application_category, application_name") if 'applications' in existing else pd.DataFrame()
    zones = safe_query(
        "SELECT zone_id, zone_name, zone_category FROM agro_zones ORDER BY zone_category, zone_name") if 'agro_zones' in existing else pd.DataFrame()
    operations = safe_query(
        "SELECT operation_id, operation_name, operation_category FROM operations ORDER BY operation_category, operation_name") if 'operations' in existing else pd.DataFrame()
    power_types = safe_query(
        "SELECT DISTINCT power_type FROM bpla_models WHERE power_type IS NOT NULL ORDER BY power_type") if 'bpla_models' in existing else pd.DataFrame()
    statuses = safe_query(
        "SELECT status_id, status_name FROM statuses ORDER BY status_name") if 'statuses' in existing else pd.DataFrame()
    suppliers = safe_query(
        "SELECT supplier_id, supplier_name FROM suppliers ORDER BY supplier_name") if 'suppliers' in existing else pd.DataFrame()

    # Диапазоны числовых полей
    ranges = {}
    if 'bpla_models' in existing:
        for col in ['payload_capacity_kg', 'takeoff_mass_kg', 'flight_time_min', 'max_speed_kmh', 'max_range_km',
                    'max_altitude_m',
                    'wingspan_or_rotor_m', 'length_m', 'battery_capacity_kwh', 'fuel_consumption_l_per_h',
                    'wind_resistance_ms', 'labor_cost_per_ha', 'energy_cost_per_ha_min', 'energy_cost_per_ha_max',
                    'recommended_field_area_min_ha', 'recommended_field_area_max_ha']:
            try:
                r = pd.read_sql_query(
                    f"SELECT MIN({col}) as min, MAX({col}) as max FROM bpla_models WHERE {col} IS NOT NULL", conn)
                ranges[col] = (r.iloc[0]['min'], r.iloc[0]['max'])
            except Exception:
                ranges[col] = (None, None)

    conn.close()
    return types, countries, manufacturers, applications, zones, operations, power_types, statuses, suppliers, ranges


# ==================== ЗАГРУЗКА ====================
with st.spinner("Загрузка данных..."):
    try:
        df, df_prices, df_apps, df_zones, df_ops, df_sensors, df_reqs, df_tco, df_type_reqs, df_comparison = load_data(
            str(db_file))
        types, countries, manufacturers, applications, zones, operations, power_types, statuses, suppliers, ranges = get_filter_options(
            str(db_file))
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {e}")
        st.stop()

# ==================== ЗАГОЛОВОК ====================
st.title("🚁 База данных БПЛА для АПК — Полная интеграция")
st.caption(f"Загружено **{len(df)}** моделей из БД: `{db_file}`")

# ==================== ФИЛЬТРЫ (sidebar) ====================
with st.sidebar:
    st.header("🔍 Расширенные фильтры")

    search = st.text_input("Поиск по названию/производителю/применению", placeholder="DJI, опрыскивание...")

    st.subheader("Тип БПЛА")
    sel_types = st.multiselect("Тип", types['type_name'].tolist() if not types.empty else [], placeholder="Все")

    if not types.empty and any(
            c in types.columns for c in ['vtol_capability', 'fixed_wing', 'rotary_wing', 'multirotor']):
        st.caption("Конструкция:")
        c1, c2 = st.columns(2)
        with c1:
            sel_vtol = st.toggle("VTOL", value=False)
            sel_fixed = st.toggle("Fixed-wing", value=False)
        with c2:
            sel_rotary = st.toggle("Вертолёт", value=False)
            sel_multi = st.toggle("Мультиротор", value=False)

    st.subheader("Производитель")
    sel_countries = st.multiselect("Страна", countries['country_name'].tolist() if not countries.empty else [],
                                   placeholder="Все")
    if sel_countries and not manufacturers.empty:
        cids = countries[countries['country_name'].isin(sel_countries)]['country_id'].tolist()
        man_opts = manufacturers[manufacturers['country_id'].isin(cids)]['manufacturer_name'].tolist()
    else:
        man_opts = manufacturers['manufacturer_name'].tolist() if not manufacturers.empty else []
    sel_manufacturers = st.multiselect("Производитель", man_opts, placeholder="Все")

    st.subheader("Применение")
    sel_apps = []
    if not applications.empty:
        for cat in applications['application_category'].dropna().unique():
            apps = applications[applications['application_category'] == cat]['application_name'].tolist()
            with st.expander(f"{cat} ({len(apps)})"):
                picked = st.multiselect(f"_{cat}", apps, label_visibility="collapsed", key=f"app_{cat}")
                sel_apps.extend(picked)

    st.subheader("Агро-зоны")
    sel_zones_list = []
    if not zones.empty:
        for cat in zones['zone_category'].dropna().unique():
            zlist = zones[zones['zone_category'] == cat]['zone_name'].tolist()
            with st.expander(f"{cat} ({len(zlist)})"):
                picked = st.multiselect(f"_{cat}", zlist, label_visibility="collapsed", key=f"zone_{cat}")
                sel_zones_list.extend(picked)

    st.subheader("Операции")
    sel_ops_list = []
    if not operations.empty:
        for cat in operations['operation_category'].dropna().unique():
            olist = operations[operations['operation_category'] == cat]['operation_name'].tolist()
            with st.expander(f"{cat} ({len(olist)})"):
                picked = st.multiselect(f"_{cat}", olist, label_visibility="collapsed", key=f"op_{cat}")
                sel_ops_list.extend(picked)

    st.subheader("Технические параметры")
    pwr_opts = power_types['power_type'].tolist() if not power_types.empty else []
    sel_power = st.multiselect("Двигатель", pwr_opts, placeholder="Все")


    def num_filter(label, col, step=1.0, default_max=5000.0):
        rmin, rmax = ranges.get(col, (0, default_max))
        if rmin is None or rmax is None or pd.isna(rmin) or pd.isna(rmax):
            rmin, rmax = 0.0, default_max
        c1, c2 = st.columns(2)
        vmin = c1.number_input(f"{label} от", float(rmin), float(rmax), float(rmin), step=step, key=f"min_{col}")
        vmax = c2.number_input(f"{label} до", float(rmin), float(rmax), float(rmax), step=step, key=f"max_{col}")
        return vmin, vmax


    p_min, p_max = num_filter("Грузоподъёмность, кг", "payload_capacity_kg", 1.0, 5000.0)
    m_min, m_max = num_filter("Взлётная масса, кг", "takeoff_mass_kg", 1.0, 5000.0)
    f_min, f_max = num_filter("Время полёта, мин", "flight_time_min", 5.0, 3000.0)
    s_min, s_max = num_filter("Макс. скорость, км/ч", "max_speed_kmh", 5.0, 500.0)
    rng_min, rng_max = num_filter("Дальность, км", "max_range_km", 10.0, 10000.0)
    a_min, a_max = num_filter("Высота, м", "max_altitude_m", 100.0, 10000.0)
    w_min, w_max = num_filter("Размах/ротор, м", "wingspan_or_rotor_m", 0.5, 50.0)
    l_min, l_max = num_filter("Длина, м", "length_m", 0.5, 50.0)
    wnd_min = st.slider("Ветроустойчивость от, м/с", 0.0, 30.0, 0.0, step=1.0)

    st.subheader("Эксплуатационные параметры")
    lc_min, lc_max = num_filter("Трудозатраты, чел.ч/га", "labor_cost_per_ha", 0.001, 1.0)
    ec_min, ec_max = num_filter("Энергозатраты мин, МДж/га", "energy_cost_per_ha_min", 1.0, 500.0)
    ec2_min, ec2_max = num_filter("Энергозатраты макс, МДж/га", "energy_cost_per_ha_max", 1.0, 500.0)

    st.subheader("Цена, ₽")
    c1, c2 = st.columns(2)
    price_min = c1.number_input("От", 0, 100_000_000, 0, step=100_000)
    price_max = c2.number_input("До", 0, 100_000_000, 100_000_000, step=100_000)

    st.subheader("Площадь поля, га")
    c1, c2 = st.columns(2)
    fa_min = c1.number_input("Площадь от", 0.0, 5000.0, 0.0, step=1.0)
    fa_max = c2.number_input("Площадь до", 0.0, 5000.0, 5000.0, step=1.0)

    st.subheader("Статус")
    sel_statuses = st.multiselect("Статус", statuses['status_name'].tolist() if not statuses.empty else [],
                                  placeholder="Все")

    if st.button("🔄 Сбросить фильтры", width='stretch'):
        st.rerun()

# ==================== ПРИМЕНЕНИЕ ФИЛЬТРОВ ====================
mask = pd.Series([True] * len(df), index=df.index)

if search:
    s = search.lower()
    mask &= (df['model_name'].str.lower().str.contains(s, na=False) |
             df['manufacturer_name'].str.lower().str.contains(s, na=False) |
             df['applications'].str.lower().str.contains(s, na=False) |
             df['operations'].str.lower().str.contains(s, na=False))

if sel_types: mask &= df['type_name'].isin(sel_types)
if not types.empty:
    if 'sel_vtol' in locals() and sel_vtol and 'vtol_capability' in df.columns: mask &= (df['vtol_capability'] == 1)
    if 'sel_fixed' in locals() and sel_fixed and 'fixed_wing' in df.columns: mask &= (df['fixed_wing'] == 1)
    if 'sel_rotary' in locals() and sel_rotary and 'rotary_wing' in df.columns: mask &= (df['rotary_wing'] == 1)
    if 'sel_multi' in locals() and sel_multi and 'multirotor' in df.columns: mask &= (df['multirotor'] == 1)

if sel_countries: mask &= df['country_name'].isin(sel_countries)
if sel_manufacturers: mask &= df['manufacturer_name'].isin(sel_manufacturers)
if sel_power: mask &= df['power_type'].isin(sel_power)
if sel_statuses: mask &= df['status_name'].isin(sel_statuses)
if sel_apps: mask &= df['applications'].apply(lambda x: any(a in str(x) for a in sel_apps))
if sel_zones_list: mask &= df['agro_zones'].apply(lambda x: any(z in str(x) for z in sel_zones_list))
if sel_ops_list: mask &= df['operations'].apply(lambda x: any(o in str(x) for o in sel_ops_list))

mask &= (df['payload_capacity_kg'].isna() | df['payload_capacity_kg'].between(p_min, p_max))
mask &= (df['takeoff_mass_kg'].isna() | df['takeoff_mass_kg'].between(m_min, m_max))
mask &= (df['flight_time_min'].isna() | df['flight_time_min'].between(f_min, f_max))
mask &= (df['max_speed_kmh'].isna() | df['max_speed_kmh'].between(s_min, s_max))
mask &= (df['max_range_km'].isna() | df['max_range_km'].between(rng_min, rng_max))
mask &= (df['max_altitude_m'].isna() | df['max_altitude_m'].between(a_min, a_max))
mask &= (df['wingspan_or_rotor_m'].isna() | df['wingspan_or_rotor_m'].between(w_min, w_max))
mask &= (df['length_m'].isna() | df['length_m'].between(l_min, l_max))
mask &= (df['wind_resistance_ms'].isna() | (df['wind_resistance_ms'] >= wnd_min))
mask &= (df['labor_cost_per_ha'].isna() | df['labor_cost_per_ha'].between(lc_min, lc_max))
mask &= (df['energy_cost_per_ha_min'].isna() | df['energy_cost_per_ha_min'].between(ec_min, ec_max))
mask &= (df['energy_cost_per_ha_max'].isna() | df['energy_cost_per_ha_max'].between(ec2_min, ec2_max))
mask &= (df['recommended_field_area_max_ha'].isna() | (df['recommended_field_area_max_ha'] >= fa_min))
mask &= (df['recommended_field_area_min_ha'].isna() | (df['recommended_field_area_min_ha'] <= fa_max))

if 'min_price' in df.columns:
    mask &= df['min_price'].fillna(0).between(price_min, price_max) | df['min_price'].isna()

df_f = df[mask].copy()

# ==================== МЕТРИКИ ====================
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Найдено", len(df_f))
avg_p = df_f['avg_price'].mean() if 'avg_price' in df_f.columns else None
c2.metric("Средняя цена", f"{avg_p:,.0f} ₽".replace(",", " ") if pd.notna(avg_p) else "—")
c3.metric("Стран", df_f['country_name'].nunique())
mp = df_f['payload_capacity_kg'].max()
c4.metric("Макс. груз", f"{mp:.1f} кг" if pd.notna(mp) else "—")
c5.metric("Макс. время", f"{df_f['flight_time_min'].max():.0f} мин" if pd.notna(df_f['flight_time_min'].max()) else "—")
c6.metric("Макс. скорость", f"{df_f['max_speed_kmh'].max():.0f} км/ч" if pd.notna(df_f['max_speed_kmh'].max()) else "—")

# ==================== ТАБЫ ====================

# ==================== ТАБЫ ====================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📋 Таблица", "📊 Графики", "🔍 Сравнение", "💰 TCO", "📋 Требования к типам", "⚖️ Сравнение типов", "🔄 Сбор данных"
])

with tab1:
    if len(df_f) == 0:
        st.warning("Ничего не найдено. Попробуйте ослабить фильтры или добавьте данные во вкладке 🔄 Сбор данных.")
    else:
        # --- Формирование таблицы с обязательными параметрами ---
        disp_cols = [
            'model_name', 'manufacturer_name', 'country_name', 'type_name',
            'payload_capacity_kg', 'takeoff_mass_kg', 'max_altitude_m', 'flight_time_min',
            'labor_cost_per_ha', 'energy_cost_per_ha_min', 'energy_cost_per_ha_max',
            'recommended_field_area_min_ha', 'recommended_field_area_max_ha',
            'agro_zones', 'wind_resistance_ms', 'ip_rating', 'temp_range_min_c', 'temp_range_max_c',
            'power_type', 'applications', 'min_price'
        ]
        disp_cols = [c for c in disp_cols if c in df_f.columns]
        disp = df_f[disp_cols].copy()

        # --- Составная колонка: стойкость к внешним воздействиям (ДО переименования и форматирования) ---
        parts_list = []
        for idx, r in disp.iterrows():
            parts = []
            # IP
            ip = r.get('ip_rating')
            if pd.notna(ip) and str(ip).strip() not in ('', 'None', '—'):
                parts.append(f"IP{ip}")
            # Температура
            t_min = r.get('temp_range_min_c')
            t_max = r.get('temp_range_max_c')
            if pd.notna(t_min) and pd.notna(t_max):
                parts.append(f"{t_min:.0f}…{t_max:.0f}°C")
            elif pd.notna(t_min):
                parts.append(f"от {t_min:.0f}°C")
            elif pd.notna(t_max):
                parts.append(f"до {t_max:.0f}°C")
            # Ветер
            wnd = r.get('wind_resistance_ms')
            if pd.notna(wnd):
                parts.append(f"ветер до {wnd:.0f} м/с")
            # Собираем
            parts_list.append("; ".join(parts) if parts else "—")
        disp['Стойкость к внешним воздействиям'] = parts_list

        # --- Переименование колонок ---
        rename_map = {
            'model_name': 'Модель', 'manufacturer_name': 'Производитель', 'country_name': 'Страна',
            'type_name': 'Тип', 'payload_capacity_kg': 'Полезная нагрузка, кг', 'takeoff_mass_kg': 'Взлётная масса, кг',
            'max_altitude_m': 'Высота полёта, м', 'flight_time_min': 'Время полёта, мин',
            'labor_cost_per_ha': 'Трудозатраты, чел.ч/га',
            'energy_cost_per_ha_min': 'Энергозатраты мин, МДж/га',
            'energy_cost_per_ha_max': 'Энергозатраты макс, МДж/га',
            'recommended_field_area_min_ha': 'Площадь мин, га', 'recommended_field_area_max_ha': 'Площадь макс, га',
            'agro_zones': 'Агротехнические зоны', 'wind_resistance_ms': 'Ветроустойчивость, м/с',
            'ip_rating': 'Защита IP', 'temp_range_min_c': 'Темп. мин, °C', 'temp_range_max_c': 'Темп. макс, °C',
            'power_type': 'Двигатель', 'applications': 'Применение', 'min_price': 'Цена от, ₽'
        }
        disp = disp.rename(columns={k: v for k, v in rename_map.items() if k in disp.columns})

        # --- Заглушки для отсутствующих в БД полей ---
        disp['Радиоэлектронная защита'] = "Нет данных в БД"
        disp['Безопасность'] = "Нет данных в БД"

        # --- Финальный порядок колонок ---
        final_cols = [
            'Модель', 'Производитель', 'Страна', 'Тип',
            'Полезная нагрузка, кг', 'Взлётная масса, кг', 'Высота полёта, м',
            'Время полёта, мин', 'Трудозатраты, чел.ч/га',
            'Энергозатраты мин, МДж/га', 'Энергозатраты макс, МДж/га',
            'Радиоэлектронная защита', 'Стойкость к внешним воздействиям', 'Безопасность',
            'Площадь мин, га', 'Площадь макс, га', 'Агротехнические зоны',
            'Двигатель', 'Применение', 'Цена от, ₽'
        ]
        final_cols = [c for c in final_cols if c in disp.columns]
        disp = disp[final_cols]

        # --- Настройка отображения числовых колонок для корректной сортировки ---
        col_config = {}
        if "Полезная нагрузка, кг" in disp.columns:
            col_config["Полезная нагрузка, кг"] = st.column_config.NumberColumn(format="%.2f")
        if "Взлётная масса, кг" in disp.columns:
            col_config["Взлётная масса, кг"] = st.column_config.NumberColumn(format="%.2f")
        if "Высота полёта, м" in disp.columns:
            col_config["Высота полёта, м"] = st.column_config.NumberColumn(format="%.2f")
        if "Время полёта, мин" in disp.columns:
            col_config["Время полёта, мин"] = st.column_config.NumberColumn(format="%.2f")
        if "Трудозатраты, чел.ч/га" in disp.columns:
            col_config["Трудозатраты, чел.ч/га"] = st.column_config.NumberColumn(format="%.3f")
        if "Энергозатраты мин, МДж/га" in disp.columns:
            col_config["Энергозатраты мин, МДж/га"] = st.column_config.NumberColumn(format="%.2f")
        if "Энергозатраты макс, МДж/га" in disp.columns:
            col_config["Энергозатраты макс, МДж/га"] = st.column_config.NumberColumn(format="%.2f")
        if "Площадь мин, га" in disp.columns:
            col_config["Площадь мин, га"] = st.column_config.NumberColumn(format="%.2f")
        if "Площадь макс, га" in disp.columns:
            col_config["Площадь макс, га"] = st.column_config.NumberColumn(format="%.2f")
        if "Цена от, ₽" in disp.columns:
            col_config["Цена от, ₽"] = st.column_config.NumberColumn(format="%.0f")

        st.dataframe(
            disp,
            hide_index=True,
            width='stretch',
            height=700,
            column_config=col_config
        )
        csv = df_f.to_csv(index=False, encoding='utf-8-sig')
        st.download_button("📥 CSV полных данных", csv, "bpla_full.csv", "text/csv")

with tab2:
    if len(df_f) == 0:
        st.info("Нет данных для отображения графиков. Добавьте данные через вкладку 🔄 Сбор данных.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            tc = df_f['type_name'].value_counts().reset_index()
            tc.columns = ['Тип', 'Количество']
            fig = _apply_transparent_bg(px.pie(tc, values='Количество', names='Тип', title='Распределение по типам БПЛА',
                         color_discrete_sequence=px.colors.qualitative.Set3))
            st.plotly_chart(fig, width='stretch', config={
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'chart',
        'height': None,
        'width': None,
        'scale': 3  # High resolution (3x)
    },
    'displaylogo': False,
    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape']
})
        with c2:
            cc = df_f['country_name'].value_counts().head(10).reset_index()
            cc.columns = ['Страна', 'Количество']
            fig = _apply_transparent_bg(px.bar(cc, x='Страна', y='Количество', title='Топ-10 стран-производителей',
                         color='Количество', color_continuous_scale='Viridis'))
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, width='stretch', config={
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'chart',
        'height': None,
        'width': None,
        'scale': 3  # High resolution (3x)
    },
    'displaylogo': False,
    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape']
})

        c1, c2 = st.columns(2)
        with c1:
            fig = _apply_transparent_bg(px.scatter(df_f, x='payload_capacity_kg', y='flight_time_min',
                                     color='type_name', size='takeoff_mass_kg',
                                     hover_data=['model_name', 'manufacturer_name'],
                                     title='Грузоподъёмность vs Время полёта'))
            st.plotly_chart(fig, width='stretch', config={
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'chart',
        'height': None,
        'width': None,
        'scale': 3  # High resolution (3x)
    },
    'displaylogo': False,
    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape']
})
        with c2:
            dp = df_f[df_f['min_price'].notna()] if 'min_price' in df_f.columns else pd.DataFrame()
            if len(dp) > 0:
                fig = _apply_transparent_bg(px.scatter(dp, x='payload_capacity_kg', y='min_price',
                                 color='type_name', size='flight_time_min',
                                 hover_data=['model_name', 'manufacturer_name'],
                                 title='Цена vs Грузоподъёмность (размер=время)'))
                st.plotly_chart(fig, width='stretch', config={
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'chart',
        'height': None,
        'width': None,
        'scale': 3  # High resolution (3x)
    },
    'displaylogo': False,
    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape']
})
            else:
                st.info("Нет данных о ценах")

        c1, c2 = st.columns(2)
        with c1:
            if 'max_speed_kmh' in df_f.columns and df_f['max_speed_kmh'].notna().any():
                fig = _apply_transparent_bg(px.histogram(df_f, x='max_speed_kmh', color='type_name',
                                   title='Распределение по максимальной скорости',
                                   nbins=20))
                st.plotly_chart(fig, width='stretch', config={
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'chart',
        'height': None,
        'width': None,
        'scale': 3  # High resolution (3x)
    },
    'displaylogo': False,
    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape']
})
        with c2:
            if 'labor_cost_per_ha' in df_f.columns and df_f['labor_cost_per_ha'].notna().any():
                fig = _apply_transparent_bg(px.box(df_f, x='type_name', y='labor_cost_per_ha',
                             title='Трудозатраты по типам БПЛА',
                             color='type_name'))
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, width='stretch', config={
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'chart',
        'height': None,
        'width': None,
        'scale': 3  # High resolution (3x)
    },
    'displaylogo': False,
    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape']
})

        c1, c2 = st.columns(2)
        with c1:
            pc = df_f['power_type'].value_counts().reset_index()
            if not pc.empty:
                pc.columns = ['Двигатель', 'Количество']
                fig = _apply_transparent_bg(px.bar(pc, x='Двигатель', y='Количество', title='По типу двигателя', color='Двигатель'))
                st.plotly_chart(fig, width='stretch', config={
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'chart',
        'height': None,
        'width': None,
        'scale': 3  # High resolution (3x)
    },
    'displaylogo': False,
    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape']
})
        with c2:
            if 'status_name' in df_f.columns and not (df_f['status_name'] == '—').all():
                sc = df_f['status_name'].value_counts().reset_index()
                sc.columns = ['Статус', 'Количество']
                fig = _apply_transparent_bg(px.pie(sc, values='Количество', names='Статус', title='По статусу'))
                st.plotly_chart(fig, width='stretch', config={
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'chart',
        'height': None,
        'width': None,
        'scale': 3  # High resolution (3x)
    },
    'displaylogo': False,
    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape']
})

        # --- Операции и производительность ---
        if not df_ops.empty:
            # Фильтруем операции только для отобранных моделей
            df_ops_f = df_ops[df_ops['model_id'].isin(df_f['model_id'])].copy() if not df_ops.empty else pd.DataFrame()

            if df_ops_f.empty:
                st.info("Нет данных об операциях для отобранных моделей.")
            else:
                st.divider()
                st.subheader("🔧 Операции и производительность")

                # --- Ряд 1: Количество моделей по категориям и операциям ---
                c1, c2 = st.columns(2)
                with c1:
                    # Уникальные модели по категориям операций
                    if 'operation_category' in df_ops_f.columns and df_ops_f['operation_category'].notna().any():
                        cat_counts = df_ops_f.groupby('operation_category')['model_id'].nunique().reset_index()
                        cat_counts.columns = ['Категория', 'Моделей']
                        cat_counts = cat_counts.sort_values('Моделей', ascending=False)
                        total_models = df_f['model_id'].nunique()
                        cat_counts['Доля, %'] = (cat_counts['Моделей'] / total_models * 100).round(1)
                        fig = _apply_transparent_bg(px.bar(
                            cat_counts, x='Категория', y='Моделей',
                            title=f'Модели по категориям операций (всего моделей: {total_models})',
                            color='Моделей', color_continuous_scale='Viridis',
                            text='Доля, %'
                        ))
                        fig.update_traces(texttemplate='%{text}%', textposition='outside')
                        fig.update_layout(xaxis_tickangle=-30)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Нет данных о категориях операций")

                with c2:
                    # Уникальные модели по конкретным операциям (топ-10)
                    if 'operation_name' in df_ops_f.columns and df_ops_f['operation_name'].notna().any():
                        op_counts = df_ops_f.groupby('operation_name')['model_id'].nunique().reset_index()
                        op_counts.columns = ['Операция', 'Моделей']
                        op_counts = op_counts.sort_values('Моделей', ascending=True).tail(10)
                        total_models = df_f['model_id'].nunique()
                        op_counts['Доля, %'] = (op_counts['Моделей'] / total_models * 100).round(1)
                        fig = _apply_transparent_bg(px.bar(
                            op_counts, x='Моделей', y='Операция',
                            title=f'Топ операций по количеству моделей (всего моделей: {total_models})',
                            color='Моделей', color_continuous_scale='Plasma',
                            orientation='h', text='Доля, %'
                        ))
                        fig.update_traces(texttemplate='%{text}%', textposition='outside')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Нет данных об операциях")

                # --- Ряд 2: Heatmap типы vs операции + Средняя продуктивность ---
                c1, c2 = st.columns(2)
                with c1:
                    # Heatmap: тип БПЛА vs категория операции (уникальные модели)
                    if ('type_name' in df_f.columns and 'operation_category' in df_ops_f.columns and
                            not df_f.empty and not df_ops_f.empty):
                        # Соединяем df_ops_f с df_f для получения типа
                        ops_type = df_ops_f.merge(df_f[['model_id', 'type_name']], on='model_id', how='left')
                        if ops_type['type_name'].notna().any() and ops_type['operation_category'].notna().any():
                            heatmap_data = ops_type.groupby(['type_name', 'operation_category'])['model_id'].nunique().reset_index()
                            heatmap_data.columns = ['Тип БПЛА', 'Категория операции', 'Моделей']
                            heatmap_pivot = heatmap_data.pivot(index='Тип БПЛА', columns='Категория операции', values='Моделей').fillna(0)
                            fig = _apply_transparent_bg(px.imshow(
                                heatmap_pivot, text_auto='.0f', aspect='auto',
                                title='Модели: тип БПЛА vs категория операции',
                                color_continuous_scale='Blues'
                            ))
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Нет данных для построения heatmap")
                    else:
                        st.info("Нет данных для heatmap")

                with c2:
                    # Средняя продуктивность по операциям (усреднение по строкам — ок, т.к. это метрика операции)
                    if 'productivity_ha_per_hour' in df_ops_f.columns and df_ops_f['productivity_ha_per_hour'].notna().any():
                        prod_by_op = df_ops_f.groupby('operation_name')['productivity_ha_per_hour'].mean().reset_index()
                        prod_by_op = prod_by_op.sort_values('productivity_ha_per_hour', ascending=True)
                        fig = _apply_transparent_bg(px.bar(
                            prod_by_op, x='productivity_ha_per_hour', y='operation_name',
                            title='Средняя продуктивность по операциям, га/ч',
                            color='productivity_ha_per_hour', color_continuous_scale='Viridis',
                            orientation='h', text='productivity_ha_per_hour'
                        ))
                        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Нет данных о продуктивности")

                # --- Ряд 3: Пригодность и точность ---
                c1, c2 = st.columns(2)
                with c1:
                    if 'suitability_score' in df_ops_f.columns and df_ops_f['suitability_score'].notna().any():
                        fig = _apply_transparent_bg(px.box(
                            df_ops_f, x='operation_name', y='suitability_score',
                            title='Пригодность по операциям (suitability score)',
                            color='operation_name', points='all'
                        ))
                        fig.update_layout(xaxis_tickangle=-45, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Нет данных о пригодности")

                with c2:
                    if 'positioning_accuracy_cm' in df_ops_f.columns and df_ops_f['positioning_accuracy_cm'].notna().any():
                        fig = _apply_transparent_bg(px.box(
                            df_ops_f, x='operation_name', y='positioning_accuracy_cm',
                            title='Точность позиционирования по операциям, см',
                            color='operation_name', points='all'
                        ))
                        fig.update_layout(xaxis_tickangle=-45, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Нет данных о точности позиционирования")

                # --- Ряд 4: Scatter plots ---
                c1, c2 = st.columns(2)
                with c1:
                    if ('spray_width_m' in df_ops_f.columns and 'productivity_ha_per_hour' in df_ops_f.columns and
                            df_ops_f['spray_width_m'].notna().any() and df_ops_f['productivity_ha_per_hour'].notna().any()):
                        df_ops_scatter = df_ops_f.dropna(subset=['spray_width_m', 'productivity_ha_per_hour'])
                        if not df_ops_scatter.empty:
                            fig = _apply_transparent_bg(px.scatter(
                                df_ops_scatter, x='spray_width_m', y='productivity_ha_per_hour',
                                color='operation_name', size='suitability_score',
                                hover_data=['model_id', 'operation_category'],
                                title='Ширина захвата vs Продуктивность'
                            ))
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Нет данных для корреляции")
                    else:
                        st.info("Нет данных о ширине захвата / продуктивности")

                with c2:
                    if ('overlap_percent' in df_ops_f.columns and 'skip_percent' in df_ops_f.columns and
                            df_ops_f['overlap_percent'].notna().any() and df_ops_f['skip_percent'].notna().any()):
                        df_ops_ov = df_ops_f.dropna(subset=['overlap_percent', 'skip_percent'])
                        if not df_ops_ov.empty:
                            fig = _apply_transparent_bg(px.scatter(
                                df_ops_ov, x='overlap_percent', y='skip_percent',
                                color='operation_name', size='suitability_score',
                                hover_data=['model_id', 'operation_category'],
                                title='Перекрытие vs Пропуски (%)'
                            ))
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Нет данных для корреляции")
                    else:
                        st.info("Нет данных о перекрытии / пропусках")

                # --- Ряд 5: Норма расхода и размер капель ---
                c1, c2 = st.columns(2)
                with c1:
                    if 'application_rate_l_per_ha' in df_ops_f.columns and df_ops_f['application_rate_l_per_ha'].notna().any():
                        rate_by_op = df_ops_f.groupby('operation_name')['application_rate_l_per_ha'].mean().reset_index()
                        rate_by_op = rate_by_op.sort_values('application_rate_l_per_ha', ascending=True)
                        fig = _apply_transparent_bg(px.bar(
                            rate_by_op, x='application_rate_l_per_ha', y='operation_name',
                            title='Средняя норма расхода жидкости, л/га',
                            color='application_rate_l_per_ha', color_continuous_scale='Plasma',
                            orientation='h', text='application_rate_l_per_ha'
                        ))
                        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Нет данных о норме расхода")

                with c2:
                    if ('droplet_size_um' in df_ops_f.columns and 'liquid_capacity_l' in df_ops_f.columns and
                            df_ops_f['droplet_size_um'].notna().any() and df_ops_f['liquid_capacity_l'].notna().any()):
                        df_ops_drop = df_ops_f.dropna(subset=['droplet_size_um', 'liquid_capacity_l'])
                        if not df_ops_drop.empty:
                            fig = _apply_transparent_bg(px.scatter(
                                df_ops_drop, x='droplet_size_um', y='liquid_capacity_l',
                                color='operation_name', size='productivity_ha_per_hour',
                                hover_data=['model_id', 'operation_category'],
                                title='Размер капель vs Ёмкость бака',
                                labels={'droplet_size_um': 'Размер капель, мкм', 'liquid_capacity_l': 'Ёмкость бака, л'}
                            ))
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Нет данных для корреляции")
                    else:
                        st.info("Нет данных о размере капель / ёмкости бака")

with tab3:
    opts = df_f.apply(lambda r: f"{r['model_name']} ({r['manufacturer_name']})", axis=1).tolist()
    sel = st.multiselect("Выберите до 5 моделей для сравнения", opts, max_selections=5)
    if sel:
        names = [s.split(' (')[0] for s in sel]
        dc = df_f[df_f['model_name'].isin(names)]

        comp_cols = ['model_name', 'manufacturer_name', 'type_name', 'takeoff_mass_kg', 'payload_capacity_kg',
                     'flight_time_min', 'max_speed_kmh', 'max_range_km', 'max_altitude_m',
                     'wingspan_or_rotor_m', 'length_m', 'wind_resistance_ms', 'power_type',
                     'labor_cost_per_ha', 'energy_cost_per_ha_min', 'energy_cost_per_ha_max',
                     'recommended_field_area_min_ha', 'recommended_field_area_max_ha']
        if 'min_price' in dc.columns: comp_cols.append('min_price')
        if 'max_price' in dc.columns: comp_cols.append('max_price')

        comp = dc[[c for c in comp_cols if c in dc.columns]].set_index('model_name')
        rename_map = {
            'manufacturer_name': 'Производитель', 'type_name': 'Тип', 'takeoff_mass_kg': 'Взлётная масса, кг',
            'payload_capacity_kg': 'Груз, кг', 'flight_time_min': 'Время, мин', 'max_speed_kmh': 'Скорость, км/ч',
            'max_range_km': 'Дальность, км', 'max_altitude_m': 'Высота, м',
            'wingspan_or_rotor_m': 'Размах/ротор, м', 'length_m': 'Длина, м',
            'wind_resistance_ms': 'Ветер, м/с', 'power_type': 'Двигатель',
            'labor_cost_per_ha': 'Трудозатраты, чел.ч/га', 'energy_cost_per_ha_min': 'Энергозатраты мин, МДж/га',
            'energy_cost_per_ha_max': 'Энергозатраты макс, МДж/га',
            'recommended_field_area_min_ha': 'Площадь мин, га', 'recommended_field_area_max_ha': 'Площадь макс, га',
            'min_price': 'Цена мин, ₽', 'max_price': 'Цена макс, ₽'
        }
        comp = comp.rename(columns={k: v for k, v in rename_map.items() if k in comp.columns})
        st.table(comp.T)

        # Radar
        cats = ['Груз', 'Время', 'Ветер', 'Скорость', 'Дальность', 'Высота', 'Взлётная масса']
        fig = go.Figure()
        for _, row in dc.iterrows():
            vals = [row['payload_capacity_kg'] or 0, row['flight_time_min'] or 0,
                    row['wind_resistance_ms'] or 0, row['max_speed_kmh'] or 0,
                    row['max_range_km'] or 0, row['max_altitude_m'] or 0, row['takeoff_mass_kg'] or 0]
            maxs = [500, 1500, 20, 200, 1000, 10000, 5000]
            nv = [min(v / m * 100, 100) if m else 0 for v, m in zip(vals, maxs)]
            fig.add_trace(go.Scatterpolar(r=nv + [nv[0]], theta=cats + [cats[0]],
                                          fill='toself', name=row['model_name']))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), showlegend=True,
                          title="Сравнение (нормализовано, %)")
        st.plotly_chart(fig, width='stretch', config={
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'chart',
        'height': None,
        'width': None,
        'scale': 3  # High resolution (3x)
    },
    'displaylogo': False,
    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape']
})

        # Операции
        if not df_ops.empty and 'model_id' in dc.columns:
            st.subheader("Сравнение по операциям (продуктивность, га/ч)")
            ops_cmp = df_ops[df_ops['model_id'].isin(dc['model_id']) & df_ops['operation_name'].notna()]
            if not ops_cmp.empty:
                pivot_ops = ops_cmp.pivot_table(
                    index='operation_name', columns='model_name',
                    values='productivity_ha_per_hour', aggfunc='first'
                )
                st.dataframe(pivot_ops, width='stretch')

        # TCO сравнение
        if not df_tco.empty and 'model_id' in dc.columns:
            st.subheader("Сравнение TCO")
            tco_cmp = df_tco[df_tco['model_id'].isin(dc['model_id'])]
            if not tco_cmp.empty:
                tco_disp = tco_cmp[['model_id', 'ownership_period_years', 'initial_cost_rub', 'total_tco_rub',
                                    'cost_per_ha_rub']].copy()
                tco_disp = tco_disp.merge(dc[['model_id', 'model_name']], on='model_id')
                tco_disp = tco_disp.set_index('model_name').drop(columns='model_id')
                tco_disp.columns = ['Период, лет', 'Начальные затраты, ₽', 'Итого TCO, ₽', 'Стоимость/га, ₽']
                st.dataframe(tco_disp, width='stretch')

with tab4:
    st.subheader("💰 Совокупная стоимость владения (TCO)")
    if not df_tco.empty and 'model_id' in df.columns:
        tco_models = df_tco.merge(df[['model_id', 'model_name', 'manufacturer_name', 'type_name']], on='model_id')
        sel_tco = st.multiselect("Выберите модели для анализа TCO", tco_models['model_name'].unique().tolist())
        if sel_tco:
            tco_f = tco_models[tco_models['model_name'].isin(sel_tco)]
            fig = _apply_transparent_bg(px.bar(tco_f, x='model_name', y='total_tco_rub', color='type_name',
                         title='TCO по моделям', text='total_tco_rub'))
            fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            st.plotly_chart(fig, width='stretch', config={
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'chart',
        'height': None,
        'width': None,
        'scale': 3  # High resolution (3x)
    },
    'displaylogo': False,
    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape']
})
            fig2 = _apply_transparent_bg(px.bar(tco_f, x='model_name',
                          y=['initial_cost_rub', 'annual_maintenance_rub', 'battery_replacement_cost_rub',
                             'fuel_cost_per_hour_rub', 'operator_cost_per_hour_rub', 'insurance_cost_annual_rub'],
                          title='Структура TCO', barmode='group'))
            st.plotly_chart(fig2, width='stretch', config={
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'chart',
        'height': None,
        'width': None,
        'scale': 3  # High resolution (3x)
    },
    'displaylogo': False,
    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape']
})
            st.dataframe(tco_f, hide_index=True, width='stretch')
        else:
            st.info("Выберите модели для отображения TCO")
    else:
        st.info("В базе отсутствуют данные TCO")

with tab5:
    st.subheader("📋 Требования к типам БАС")
    if not df_type_reqs.empty:
        for tname in df_type_reqs['type_name'].dropna().unique():
            with st.expander(f"Тип: {tname}"):
                sub = df_type_reqs[df_type_reqs['type_name'] == tname]
                desc = sub['type_description'].iloc[0] if 'type_description' in sub.columns and sub[
                    'type_description'].notna().any() else ""
                if desc:
                    st.caption(desc)
                # Группировка по категории
                if 'requirement_category' in sub.columns and sub['requirement_category'].notna().any():
                    for cat in sub['requirement_category'].unique():
                        st.markdown(f"**{cat}**")
                        cat_sub = sub[sub['requirement_category'] == cat]
                        disp = cat_sub[
                            ['parameter_name', 'parameter_value', 'parameter_unit', 'perspective_year']].copy()
                        disp.columns = ['Параметр', 'Значение', 'Ед. изм.', 'Целевой год']
                        st.dataframe(disp, hide_index=True, width='stretch')
                else:
                    disp = sub[['parameter_name', 'parameter_value', 'parameter_unit', 'perspective_year']].copy()
                    disp.columns = ['Параметр', 'Значение', 'Ед. изм.', 'Целевой год']
                    st.dataframe(disp, hide_index=True, width='stretch')
    else:
        st.info("В базе отсутствуют требования к типам (таблица type_requirements)")
        st.markdown("""
                **Рекомендуемые категории требований для внесения в БД:**
                - Аэродинамические и летно-технические характеристики
                - Габариты и массовые параметры
                - Силовая установка и энергетика
                - Навигация и позиционирование
                - Радиоэлектронная защита (подавление помех, защита от перехвата)
                - Стойкость к внешним воздействиям (температура, влага, пыль, химические агрессивные среды)
                - Безопасность (аварийные системы, парашюты, геозоны, защита от столкновений)
                - Эргономика и трудозатраты
                - Экологические требования
                """)

with tab6:
    st.subheader("⚖️ Сравнительные матрицы типов БАС")
    if not df_comparison.empty:
        st.dataframe(df_comparison[
            ['matrix_name', 'type_1_name', 'type_2_name', 'comparison_criteria', 'winner_name', 'notes']].rename(
            columns={
                'matrix_name': 'Критерий сравнения', 'type_1_name': 'Тип 1', 'type_2_name': 'Тип 2',
                'comparison_criteria': 'Параметры сравнения', 'winner_name': 'Предпочтительный тип',
                'notes': 'Примечания'
            }), hide_index=True, width='stretch')
    else:
        st.info("В базе отсутствуют сравнительные матрицы (таблица comparison_matrices)")
        st.markdown("""
                **Рекомендуемые направления сравнения типов:**
                | Критерий | Мультиротор | Самолётный | Вертолётный | Гибридный (VTOL) |
                |---|---|---|---|---|
                | Взлётная масса | до 200 кг | до 1500 кг | до 600 кг | до 600 кг |
                | Грузоподъёмность | до 250 кг | до 450 кг | до 230 кг | до 200 кг |
                | Время полёта | до 55 мин | до 36 ч | до 12 ч | до 13 ч |
                | Скорость | низкая | высокая | средняя | высокая |
                | Площадь поля | до 500 га | до 5000 га | до 1000 га | до 1000 га |
                | Сложность рельефа | любая | ровный | любая | любая |
                | Точность опрыскивания | высокая | средняя | высокая | высокая |
                | Стоимость/га | средняя | низкая | высокая | средняя |
                """)

with tab7:
    if ingestor is None:
        st.error("❌ Модуль записи в БД не инициализирован. Создайте базу данных.")
    else:
        st.header("🔄 Сбор и импорт данных о БПЛА")
        st.caption("Автоматический веб-сбор, импорт CSV и ручной ввод — всё записывается в базу данных")

        subtab1, subtab2, subtab3, subtab4 = st.tabs([
            "🌐 Автосбор с сайтов", "📁 Импорт CSV/Excel", "✏️ Ручной ввод", "📋 Журнал"
        ])

        with subtab1:
            st.subheader("🌐 Автоматический сбор с любого сайта")

            st.info("""
            Введите URL страницы с характеристиками БПЛА. Система автоматически проанализирует структуру страницы,
            найдёт таблицы, списки и блоки с техническими параметрами, определит соответствие полей и извлечёт данные.

            **Поддерживаемые структуры:** HTML-таблицы, списки определений (dl/dt/dd), div-блоки с параметрами.

            **Примеры URL:**
            - `https://www.dji.com/t50` — страницы производителей
            - `https://www.xa.com/p100` — каталоги продукции
            - `https://example.com/drones/compare` — таблицы сравнения
            """)

            url_input = st.text_input("URL страницы", placeholder="https://www.dji.com/t50")

            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                update_existing = st.toggle("Обновлять существующие модели", value=False)
            with col_opt2:
                preview_before_save = st.toggle("Предпросмотр перед записью", value=True)

            if st.button("🚀 Собрать данные", type="primary", width='stretch'):
                if not url_input or not url_input.startswith('http'):
                    st.error("Введите корректный URL, начинающийся с http:// или https://")
                else:
                    with st.spinner("Анализ страницы и сбор данных..."):
                        scraper = SmartScraper(cache_dir="./scraping_cache")
                        results = scraper.scrape(url_input)

                    if not results:
                        st.warning("Не удалось извлечь данные с указанной страницы. Попробуйте другой URL.")
                    else:
                        st.success(f"Найдено {len(results)} моделей!")

                        preview_data = []
                        for r in results:
                            preview_data.append({
                                'Модель': r.model_name,
                                'Производитель': r.manufacturer_name,
                                'Груз, кг': r.payload_capacity_kg,
                                'Время, мин': r.flight_time_min,
                                'Скорость, км/ч': r.max_speed_kmh,
                                'Высота, м': r.max_altitude_m,
                                'Ветер, м/с': r.wind_resistance_ms,
                            })
                        preview_df = pd.DataFrame(preview_data)
                        st.dataframe(preview_df, width='stretch', hide_index=True)

                        if preview_before_save:
                            st.markdown("**Выберите модели для записи в БД:**")
                            selected_models = st.multiselect(
                                "Модели",
                                options=[f"{r.model_name} ({r.manufacturer_name})" for r in results],
                                default=[f"{r.model_name} ({r.manufacturer_name})" for r in results],
                                key="web_select_models"
                            )

                            if st.button("💾 Записать в БД", type="primary", width='stretch'):
                                selected_names = [s.split(" (")[0] for s in selected_models]
                                filtered_results = [r for r in results if r.model_name in selected_names]

                                success = 0
                                skipped = 0
                                errors = []

                                for uav in filtered_results:
                                    try:
                                        model = uav.to_uav_model_raw()
                                        ok, msg = ingestor.ingest(model, source_name=uav.source_site,
                                                                  update_existing=update_existing)
                                        if ok:
                                            success += 1
                                        else:
                                            if "Дубликат" in msg:
                                                skipped += 1
                                            else:
                                                errors.append(f"{uav.model_name}: {msg}")
                                    except Exception as e:
                                        errors.append(f"{uav.model_name}: {str(e)}")

                                c1, c2, c3 = st.columns(3)
                                c1.metric("✅ В БД", success)
                                c2.metric("⏭ Дубликаты", skipped)
                                c3.metric("❌ Ошибок", len(errors))

                                if errors:
                                    with st.expander(f"❌ Ошибки ({len(errors)})"):
                                        for e in errors[:20]:
                                            st.error(e)

                                if success > 0:
                                    st.success(f"Записано {success} моделей!")
                                    st.button("🔄 Перезагрузить", on_click=lambda: st.rerun(), key="reload_web1")
                        else:
                            success = 0
                            skipped = 0
                            errors = []

                            for uav in results:
                                try:
                                    model = uav.to_uav_model_raw()
                                    ok, msg = ingestor.ingest(model, source_name=uav.source_site,
                                                              update_existing=update_existing)
                                    if ok:
                                        success += 1
                                    else:
                                        if "Дубликат" in msg:
                                            skipped += 1
                                        else:
                                            errors.append(f"{uav.model_name}: {msg}")
                                except Exception as e:
                                    errors.append(f"{uav.model_name}: {str(e)}")

                            st.subheader("📊 Результаты")
                            c1, c2, c3 = st.columns(3)
                            c1.metric("🔍 Собрано", len(results))
                            c2.metric("✅ В БД", success)
                            c3.metric("❌ Ошибок", len(errors))

                            if errors:
                                with st.expander("Ошибки"):
                                    for e in errors[:20]:
                                        st.error(e)

                            if success > 0:
                                st.success(f"Успешно записано {success} моделей!")
                                st.button("🔄 Перезагрузить", on_click=lambda: st.rerun(), key="reload_web2")

        with subtab2:
            st.subheader("Импорт данных из CSV/Excel")
            st.info("Загрузите CSV-файл с характеристиками БПЛА. Система автоматически сопоставит колонки с полями БД.")

            uploaded_file = st.file_uploader("Выберите CSV-файл", type=["csv"])

            if uploaded_file is not None:
                df_upload = pd.read_csv(uploaded_file, encoding="utf-8-sig")
                st.caption(f"Загружено **{len(df_upload)}** строк, **{len(df_upload.columns)}** колонок")
                st.dataframe(df_upload.head(5), width='stretch')

                st.subheader("Настройка маппинга колонок")
                st.caption("Укажите, какие колонки CSV соответствуют полям базы данных")

                csv_cols = ["(пропустить)"] + list(df_upload.columns)
                default_mapping = get_default_csv_mapping()

                user_mapping = {}
                col_left, col_right = st.columns(2)

                fields = list(default_mapping.keys())
                half = len(fields) // 2

                for i, field in enumerate(fields):
                    with col_left if i < half else col_right:
                        default_idx = 0
                        if default_mapping[field] in csv_cols:
                            default_idx = csv_cols.index(default_mapping[field])
                        selected = st.selectbox(
                            f"{field}", csv_cols, index=default_idx, key=f"csv_map_{field}"
                        )
                        if selected != "(пропустить)":
                            user_mapping[field] = selected

                update_existing_csv = st.toggle("Обновлять существующие модели", value=False, key="upd_csv")

                if st.button("📥 Импортировать в БД", type="primary", width='stretch'):
                    tmp_path = Path(tempfile.gettempdir()) / uploaded_file.name
                    df_upload.to_csv(tmp_path, index=False, encoding="utf-8-sig")

                    with st.spinner("Импорт..."):
                        success, skipped, errors = ingestor.import_csv(
                            str(tmp_path), user_mapping, source_name="csv_import"
                        )

                    col_s, col_sk, col_e = st.columns(3)
                    col_s.metric("✅ Успешно", success)
                    col_sk.metric("⏭ Пропущено", skipped)
                    col_e.metric("❌ Ошибок", len(errors))

                    if errors:
                        with st.expander("Ошибки импорта"):
                            for e in errors[:30]:
                                st.error(e)

                    if success > 0:
                        st.success(f"Импортировано {success} моделей!")
                        st.button("🔄 Перезагрузить", on_click=lambda: st.rerun(), key="reload_csv")

        with subtab3:
            st.subheader("Ручное добавление модели БПЛА")
            st.info("Заполните форму для добавления новой модели. Обязательны только название и производитель.")

            with st.form("manual_model_form"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**Основные сведения**")
                    m_name = st.text_input("Название модели *", placeholder="Напр: Agras T40")
                    m_manufacturer = st.text_input("Производитель *", placeholder="Напр: DJI")
                    m_country = st.text_input("Страна", placeholder="Напр: Китай")
                    m_type = st.selectbox("Тип БПЛА",
                                          ["", "Мультиротор", "Самолётный", "Вертолётный", "Гибридный (VTOL)"])
                    m_status = st.text_input("Статус", value="Активная")
                    m_source = st.text_input("URL источника")
                    m_notes = st.text_area("Примечания")

                with col2:
                    st.markdown("**Технические характеристики**")
                    m_payload = st.number_input("Грузоподъёмность, кг", min_value=0.0, step=0.1)
                    m_takeoff = st.number_input("Взлётная масса, кг", min_value=0.0, step=0.1)
                    m_flight = st.number_input("Время полёта, мин", min_value=0.0, step=1.0)
                    m_speed = st.number_input("Макс. скорость, км/ч", min_value=0.0, step=1.0)
                    m_range = st.number_input("Дальность, км", min_value=0.0, step=1.0)
                    m_altitude = st.number_input("Высота, м", min_value=0.0, step=10.0)
                    m_wingspan = st.number_input("Размах/ротор, м", min_value=0.0, step=0.1)
                    m_length = st.number_input("Длина, м", min_value=0.0, step=0.1)
                    m_power = st.selectbox("Тип двигателя",
                                           ["", "Электрический", "Бензиновый", "Дизельный", "Гибридный"])
                    m_ip = st.text_input("IP-рейтинг", placeholder="Напр: 67")

                with col3:
                    st.markdown("**Эксплуатационные параметры**")
                    m_wind = st.number_input("Ветроустойчивость, м/с", min_value=0.0, step=0.5)
                    m_temp_min = st.number_input("Темп. мин, °C", value=-20.0, step=1.0)
                    m_temp_max = st.number_input("Темп. макс, °C", value=50.0, step=1.0)
                    m_labor = st.number_input("Трудозатраты, чел.ч/га", min_value=0.0, step=0.001, format="%.3f")
                    m_energy_min = st.number_input("Энергозатраты мин, МДж/га", min_value=0.0, step=0.1)
                    m_energy_max = st.number_input("Энергозатраты макс, МДж/га", min_value=0.0, step=0.1)
                    m_area_min = st.number_input("Рек. площадь мин, га", min_value=0.0, step=1.0)
                    m_area_max = st.number_input("Рек. площадь макс, га", min_value=0.0, step=1.0)
                    m_battery = st.number_input("Ёмкость батареи, кВт·ч", min_value=0.0, step=0.1)
                    m_fuel = st.number_input("Расход топлива, л/ч", min_value=0.0, step=0.1)

                st.markdown("**💰 Цена (опционально)**")
                col_p1, col_p2, col_p3 = st.columns(3)
                with col_p1:
                    m_price = st.number_input("Цена, ₽", min_value=0, step=1000)
                with col_p2:
                    m_price_type = st.selectbox("Тип цены", ["retail", "dealer", "promo"])
                with col_p3:
                    m_supplier = st.text_input("Поставщик", value="Ручной ввод")

                submitted = st.form_submit_button("💾 Сохранить в БД", width='stretch', type="primary")

            if submitted:
                if not m_name or not m_manufacturer:
                    st.error("Название модели и производитель обязательны!")
                else:
                    model = UAVModelRaw(
                        model_name=m_name.strip(),
                        manufacturer_name=m_manufacturer.strip(),
                        country_name=m_country.strip() if m_country else None,
                        type_name=m_type if m_type else None,
                        status_name=m_status.strip() if m_status else "Активная",
                        payload_capacity_kg=m_payload if m_payload > 0 else None,
                        takeoff_mass_kg=m_takeoff if m_takeoff > 0 else None,
                        flight_time_min=m_flight if m_flight > 0 else None,
                        max_speed_kmh=m_speed if m_speed > 0 else None,
                        max_range_km=m_range if m_range > 0 else None,
                        max_altitude_m=m_altitude if m_altitude > 0 else None,
                        wingspan_or_rotor_m=m_wingspan if m_wingspan > 0 else None,
                        length_m=m_length if m_length > 0 else None,
                        power_type=m_power if m_power else None,
                        ip_rating=m_ip.strip() if m_ip else None,
                        wind_resistance_ms=m_wind if m_wind > 0 else None,
                        temp_range_min_c=m_temp_min,
                        temp_range_max_c=m_temp_max,
                        labor_cost_per_ha=m_labor if m_labor > 0 else None,
                        energy_cost_per_ha_min=m_energy_min if m_energy_min > 0 else None,
                        energy_cost_per_ha_max=m_energy_max if m_energy_max > 0 else None,
                        recommended_field_area_min_ha=m_area_min if m_area_min > 0 else None,
                        recommended_field_area_max_ha=m_area_max if m_area_max > 0 else None,
                        battery_capacity_kwh=m_battery if m_battery > 0 else None,
                        fuel_consumption_l_per_h=m_fuel if m_fuel > 0 else None,
                        source_url=m_source.strip() if m_source else None,
                        notes=m_notes.strip() if m_notes else None,
                    )

                    if m_price > 0:
                        model.prices.append({
                            'price_rub': m_price,
                            'price_type': m_price_type,
                            'supplier_name': m_supplier.strip(),
                            'currency': 'RUB'
                        })

                    ok, msg = ingestor.ingest(model, source_name="manual", update_existing=False)
                    if ok:
                        st.success(f"✅ {msg}")
                        st.balloons()
                    else:
                        st.error(f"❌ {msg}")

        with subtab4:
            st.subheader("📋 Журнал операций сбора данных")
            st.caption("Просмотр истории импорта, ошибок и дубликатов")

            if ingestor:
                log_status = st.selectbox("Фильтр по статусу",
                                          ["Все", "success", "duplicate", "validation_error", "error"])
                log_limit = st.slider("Количество записей", 10, 500, 100)

                status_filter = None if log_status == "Все" else log_status
                logs_df = ingestor.get_logs(limit=log_limit, status=status_filter)

                if not logs_df.empty:
                    logs_df["collected_at"] = pd.to_datetime(logs_df["collected_at"])
                    logs_df = logs_df.sort_values("collected_at", ascending=False)

                    st.dataframe(
                        logs_df[
                            ["collected_at", "source_name", "model_name", "manufacturer_name", "status", "message"]],
                        width='stretch', hide_index=True
                    )

                    st.subheader("Статистика сборов")
                    stats = logs_df["status"].value_counts().reset_index()
                    stats.columns = ["Статус", "Количество"]
                    c1, c2 = st.columns(2)
                    with c1:
                        fig = _apply_transparent_bg(px.pie(stats, values="Количество", names="Статус", title="Распределение статусов"))
                        st.plotly_chart(fig, width='stretch', config={
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'chart',
        'height': None,
        'width': None,
        'scale': 3  # High resolution (3x)
    },
    'displaylogo': False,
    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape']
})
                    with c2:
                        daily = logs_df.groupby(logs_df["collected_at"].dt.date).size().reset_index()
                        daily.columns = ["Дата", "Количество"]
                        fig = _apply_transparent_bg(px.bar(daily, x="Дата", y="Количество", title="Активность по дням"))
                        st.plotly_chart(fig, width='stretch', config={
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'chart',
        'height': None,
        'width': None,
        'scale': 3  # High resolution (3x)
    },
    'displaylogo': False,
    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape']
})
                else:
                    st.info("Логи отсутствуют. Выполните сбор или импорт данных.")
            else:
                st.error("Инжектор данных не инициализирован")
# ==================== ДЕТАЛИ МОДЕЛЕЙ ====================
st.divider()
st.subheader("📖 Детальные карточки моделей")

for _, row in df_f.iterrows():
    mid = row['model_id']
    tags = []
    if row.get('vtol_capability') == 1: tags.append("VTOL")
    if row.get('fixed_wing') == 1: tags.append("Fixed-wing")
    if row.get('rotary_wing') == 1: tags.append("Вертолёт")
    if row.get('multirotor') == 1: tags.append("Мультиротор")
    tag_str = " | ".join(tags) if tags else ""

    with st.expander(
            f"{row['model_name']} — {row['manufacturer_name']} ({row['country_name']}){f' | {tag_str}' if tag_str else ''} | Статус: {row['status_name']}"):
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown("**🛠 Технические характеристики**")
            st.write(f"Тип: **{row['type_name']}**")
            if row.get('type_description') and str(row['type_description']) not in ('None', ''):
                st.caption(row['type_description'])
            st.write(f"Взлётная масса: {row['takeoff_mass_kg'] or '—'} кг")
            st.write(f"Грузоподъёмность: {row['payload_capacity_kg'] or '—'} кг")
            st.write(f"Время полёта: {row['flight_time_min'] or '—'} мин")
            st.write(f"Макс. скорость: {row['max_speed_kmh'] or '—'} км/ч")
            st.write(f"Дальность: {row['max_range_km'] or '—'} км")
            st.write(f"Высота: {row['max_altitude_m'] or '—'} м")
            st.write(f"Размах/ротор: {row['wingspan_or_rotor_m'] or '—'} м")
            st.write(f"Длина: {row['length_m'] or '—'} м")
            st.write(f"Двигатель: {row['power_type'] or '—'}")
            if pd.notna(row['battery_capacity_kwh']):
                st.write(f"Батарея: {row['battery_capacity_kwh']} кВт·ч")
            if pd.notna(row['fuel_consumption_l_per_h']):
                st.write(f"Расход топлива: {row['fuel_consumption_l_per_h']} л/ч")
            if row.get('ip_rating') and str(row['ip_rating']) not in ('None', ''):
                st.write(f"Защита корпуса (IP): {row['ip_rating']}")

        with c2:
            st.markdown("**🌾 Эксплуатация и агротребования**")
            st.write(f"Ветроустойчивость: {row['wind_resistance_ms'] or '—'} м/с")
            st.write(f"Температурный диапазон: {row['temp_range_min_c'] or '—'}°C … {row['temp_range_max_c'] or '—'}°C")
            st.write(f"Трудозатраты: {row['labor_cost_per_ha'] or '—'} чел.ч/га")
            st.write(
                f"Энергозатраты: {row['energy_cost_per_ha_min'] or '—'} – {row['energy_cost_per_ha_max'] or '—'} МДж/га")
            st.write(
                f"Рек. площадь: {row['recommended_field_area_min_ha'] or '—'} – {row['recommended_field_area_max_ha'] or '—'} га")

            st.markdown("**📋 Применения**")
            apps = df_apps[df_apps['model_id'] == mid] if not df_apps.empty else pd.DataFrame()
            if not apps.empty:
                for _, a in apps.iterrows():
                    primary = "⭐" if a.get('is_primary') == 1 else ""
                    score = f" — эффективность: {a['efficiency_score']}/5" if pd.notna(
                        a.get('efficiency_score')) else ""
                    st.write(f"{primary} {a['application_name']} ({a['application_category']}){score}")
            else:
                st.write("—")

            st.markdown("**🗺 Агро-зоны**")
            zones_m = df_zones[df_zones['model_id'] == mid] if not df_zones.empty else pd.DataFrame()
            if not zones_m.empty:
                for _, z in zones_m.iterrows():
                    suit = f" — пригодность: {z['suitability_score']}/5" if pd.notna(z.get('suitability_score')) else ""
                    st.write(f"{z['zone_name']} ({z['zone_category']}){suit}")
                    if z.get('terrain_complexity') and str(z['terrain_complexity']) not in ('None', ''):
                        st.caption(f"Рельеф: {z['terrain_complexity']}")
            else:
                st.write("—")

        with c3:
            st.markdown("**💰 Цены и поставщики**")
            prices = df_prices[df_prices['model_id'] == mid] if not df_prices.empty else pd.DataFrame()
            if not prices.empty:
                for _, p in prices.iterrows():
                    cfg = f" ({p['configuration']})" if p.get('configuration') else ""
                    avail = f" | {p['availability']}" if p.get('availability') else ""
                    delivery = f" | {p['delivery_time_days']} дн." if pd.notna(p.get('delivery_time_days')) else ""
                    warranty = f" | {p['warranty_months']} мес. гарантия" if pd.notna(p.get('warranty_months')) else ""
                    vat = "с НДС" if p.get('vat_included') == 1 else "без НДС"
                    st.write(f"**{p['price_rub']:,.0f} ₽** {vat}{cfg}{avail}{delivery}{warranty}")
                    if p.get('supplier_name'):
                        st.caption(f"Поставщик: {p['supplier_name']}")
                    if p.get('price_type'):
                        st.caption(f"Тип цены: {p['price_type']}")
                    if p.get('price_date'):
                        st.caption(f"Актуально на: {p['price_date']}")
            else:
                st.write("Цены не указаны")

            st.markdown("**💰 TCO (если есть)**")
            if not df_tco.empty:
                tco_m = df_tco[df_tco['model_id'] == mid]
                if not tco_m.empty:
                    for _, t in tco_m.iterrows():
                        st.write(f"Период владения: {t['ownership_period_years']} лет")
                        st.write(f"Начальные затраты: {t['initial_cost_rub']:,.0f} ₽".replace(",", " ") if pd.notna(
                            t['initial_cost_rub']) else "—")
                        st.write(f"Итого TCO: {t['total_tco_rub']:,.0f} ₽".replace(",", " ") if pd.notna(
                            t['total_tco_rub']) else "—")
                        st.write(f"Стоимость/га: {t['cost_per_ha_rub']:,.0f} ₽".replace(",", " ") if pd.notna(
                            t['cost_per_ha_rub']) else "—")
                else:
                    st.write("—")
            else:
                st.write("—")

            st.markdown("**🔗 Источник**")
            if row['source_url'] and str(row['source_url']) not in ('—', 'None', ''):
                st.markdown(f"[Открыть источник]({row['source_url']})")
            else:
                st.write("—")

        with c4:
            st.markdown("**🛡️ Стойкость и безопасность**")
            st.write(f"IP-рейтинг (пыль/влага): {row['ip_rating'] or '—'}")
            st.write(f"Температурный режим: {row['temp_range_min_c'] or '—'} … {row['temp_range_max_c'] or '—'} °C")
            st.write(f"Ветроустойчивость: {row['wind_resistance_ms'] or '—'} м/с")
            st.caption(
                "*Радиоэлектронная защита, системы безопасности (парашют, геозоны, обход препятствий) — требуется дополнение БД*")

            st.markdown("**📷 Совместимые сенсоры**")
            sensors = df_sensors[df_sensors['model_id'] == mid] if not df_sensors.empty else pd.DataFrame()
            if not sensors.empty:
                sens_cols = [c for c in sensors.columns if c not in ('model_id', 'sensor_id', 'is_compatible')]
                sens_disp = sensors[sens_cols].copy()
                rename_sens = {
                    'sensor_name': 'Сенсор', 'sensor_type': 'Тип', 'resolution_mp': 'Разрешение, Мп',
                    'spectral_bands': 'Спектральные каналы', 'weight_kg': 'Масса, кг',
                    'power_consumption_w': 'Потребление, Вт', 'gimbal_stabilization': 'Стабилизация',
                    'is_default': 'Штатный', 'max_flight_time_with_sensor_min': 'Время полёта, мин',
                    'notes': 'Примечания'
                }
                sens_disp = sens_disp.rename(columns={k: v for k, v in rename_sens.items() if k in sens_disp.columns})
                if 'Стабилизация' in sens_disp.columns:
                    sens_disp['Стабилизация'] = sens_disp['Стабилизация'].apply(
                        lambda x: 'Да' if x == 1 else ('Нет' if pd.notna(x) else '—'))
                if 'Штатный' in sens_disp.columns:
                    sens_disp['Штатный'] = sens_disp['Штатный'].apply(
                        lambda x: 'Да' if x == 1 else ('Нет' if pd.notna(x) else '—'))
                st.dataframe(sens_disp, hide_index=True, width='stretch')
            else:
                st.write("Нет данных по сенсорам")

        # --- Операции ---
        st.markdown("**🔧 Операции и производительность**")
        ops = df_ops[df_ops['model_id'] == mid] if not df_ops.empty else pd.DataFrame()
        if not ops.empty:
            ops_disp_cols = ['operation_name', 'operation_category', 'productivity_ha_per_hour',
                             'spray_width_m', 'liquid_capacity_l', 'droplet_size_um',
                             'application_rate_l_per_ha', 'positioning_accuracy_cm',
                             'overlap_percent', 'skip_percent', 'suitability_score', 'notes']
            ops_disp_cols = [c for c in ops_disp_cols if c in ops.columns]
            ops_disp = ops[ops_disp_cols].copy()
            rename_ops = {
                'operation_name': 'Операция', 'operation_category': 'Категория',
                'productivity_ha_per_hour': 'Продуктивность, га/ч',
                'spray_width_m': 'Ширина захвата, м', 'liquid_capacity_l': 'Ёмкость бака, л',
                'droplet_size_um': 'Размер капель, мкм',
                'application_rate_l_per_ha': 'Норма расхода, л/га', 'positioning_accuracy_cm': 'Точность поз., см',
                'overlap_percent': 'Перекрытие, %', 'skip_percent': 'Пропуски, %', 'suitability_score': 'Пригодность',
                'notes': 'Примечания'
            }
            ops_disp = ops_disp.rename(columns={k: v for k, v in rename_ops.items() if k in ops_disp.columns})
            st.dataframe(ops_disp, hide_index=True, width='stretch')

            # Агротребования
            if not df_reqs.empty:
                op_ids = ops['operation_id'].unique()
                reqs = df_reqs[df_reqs['operation_id'].isin(op_ids)]
                if not reqs.empty:
                    st.markdown("**📋 Агротребования по операциям**")
                    req_cols = ['req_operation_name', 'crop_type', 'growth_stage',
                                'min_flight_height_m', 'max_flight_height_m', 'max_wind_speed_ms',
                                'max_temperature_c', 'min_temperature_c', 'buffer_zone_m',
                                'required_overlap_percent', 'notes']
                    req_cols = [c for c in req_cols if c in reqs.columns]
                    req_disp = reqs[req_cols].copy()
                    rename_req = {
                        'req_operation_name': 'Операция', 'crop_type': 'Культура', 'growth_stage': 'Фаза',
                        'min_flight_height_m': 'Мин. высота, м', 'max_flight_height_m': 'Макс. высота, м',
                        'max_wind_speed_ms': 'Макс. ветер, м/с', 'max_temperature_c': 'Макс. темп, °C',
                        'min_temperature_c': 'Мин. темп, °C', 'buffer_zone_m': 'Буферная зона, м',
                        'required_overlap_percent': 'Перекрытие, %', 'notes': 'Примечания'
                    }
                    req_disp = req_disp.rename(columns={k: v for k, v in rename_req.items() if k in req_disp.columns})
                    st.dataframe(req_disp, hide_index=True, width='stretch')
        else:
            st.write("Нет данных по операциям")

        if row.get('notes') and str(row['notes']) not in ('None', ''):
            st.markdown("**📝 Примечания**")
            st.write(row['notes'])