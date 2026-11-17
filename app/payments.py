"""Sandbox payment gateway (test mode only - no real money, no network calls).

The interface mirrors what you would implement for Stripe or PayMongo:
    gateway.charge(method, card_number, amount) -> ChargeResult
To go live, add a StripeGateway / PayMongoGateway class with the same method
and select it with PAYMENT_PROVIDER. Card numbers are NEVER stored - only the
last 4 digits and the provider reference.
"""
import secrets
from dataclasses import dataclass

TEST_CARDS = {
    "4242424242424242": (True, "Approved"),
    "4000000000000002": (False, "Card declined"),
    "4000000000009995": (False, "Insufficient funds"),
}


@dataclass
class ChargeResult:
    ok: bool
    message: str
    ref: str = ""
    last4: str = ""


class SandboxGateway:
    name = "sandbox"

    def charge(self, method: str, card_number: str, amount: int) -> ChargeResult:
        ref = "sbx_" + secrets.token_hex(6)
        if method == "gcash":  # simulated e-wallet redirect that always approves
            return ChargeResult(True, "GCash (sandbox) approved", ref, "GCSH")
        digits = "".join(ch for ch in card_number if ch.isdigit())
        if len(digits) < 12:
            return ChargeResult(False, "Invalid card number")
        ok, msg = TEST_CARDS.get(digits, (False, "Unknown test card - use 4242 4242 4242 4242"))
        return ChargeResult(ok, msg, ref if ok else "", digits[-4:])


gateway = SandboxGateway()
