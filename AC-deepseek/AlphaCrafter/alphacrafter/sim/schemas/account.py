from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from .order import OrderSchema
from enum import Enum

class PositionDirection(str, Enum):
    """Position direction enumeration"""
    LONG = "LONG"
    SHORT = "SHORT"

class PositionData(BaseModel):
    """Position information"""
    symbol: str = Field(..., description="Stock code")
    direction: PositionDirection = Field(default=PositionDirection.LONG, description="Position direction (long/short)")
    # The benchmark online contract permits fractional units.  Keep these as
    # floats all the way through Pydantic and account.json; truncating here
    # would make a full-investment 15-asset allocation impossible.
    quantity: float = Field(..., description="Position quantity (fractional units allowed online)")
    available_quantity: float = Field(..., description="Available quantity (fractional units allowed online)")
    cost_price: float = Field(..., description="Cost price")
    current_price: float = Field(..., description="Current price")
    market_value: float = Field(..., description="Market value")
    profit_loss: float = Field(..., description="Floating profit/loss")
    profit_loss_rate: float = Field(..., description="Profit/loss ratio")

class AccountSchema(BaseModel):
    """Account related data - all parts that can be controlled"""
    
    initial_capital: float = Field(default=100_000_000.0, description="Initial capital used as the P&L baseline")
    total_assets: float = Field(..., description="Total assets")
    net_assets: float = Field(..., description="Net assets")
    available_cash: float = Field(..., description="Available cash")
    market_value: float = Field(..., description="Position market value")
    total_profit_loss: float = Field(..., description="Total floating profit/loss")
    total_profit_loss_rate: float = Field(..., description="Total profit/loss ratio")
    gross_position_rate: float = Field(..., description="Position ratio (based on absolute value)")
    net_position_rate: float = Field(..., description="Net position ratio (considering long and short offset)")
    positions: List[PositionData] = Field(default_factory=list, description="Position information")
    orders: List[OrderSchema] = Field(default_factory=list, description="Current order list")
    watch_list: List[str] = Field(default_factory=list, description="Watch list")
