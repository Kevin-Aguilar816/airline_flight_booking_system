"""E-ticket PDF generation (ReportLab) - one boarding-pass page per passenger."""
from io import BytesIO

from reportlab.graphics import renderPDF
from reportlab.graphics.barcode import qr
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas


def build_ticket_pdf(booking) -> bytes:
    f = booking.flight
    buf = BytesIO()
    W, H = landscape(A4)
    c = canvas.Canvas(buf, pagesize=(W, H))
    c.setTitle(f"E-Ticket {booking.ref}")
    for p in booking.passengers:
        c.setFillColor(colors.HexColor("#0f3d73"))
        c.rect(0, H - 90, W, 90, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 28)
        c.drawString(40, H - 55, "Flight In, Flight Out")
        c.setFont("Helvetica", 12)
        c.drawRightString(W - 40, H - 45, "BOARDING PASS / E-TICKET")
        c.drawRightString(W - 40, H - 62, f"Booking reference: {booking.ref}")

        c.setFillColor(colors.black)
        c.setFont("Helvetica", 11)
        c.drawString(40, H - 135, "PASSENGER")
        c.setFont("Helvetica-Bold", 22)
        c.drawString(40, H - 160, p.name.upper())

        c.setFont("Helvetica-Bold", 54)
        c.drawString(40, H - 250, f.origin)
        c.drawString(260, H - 250, f.dest)
        c.setFont("Helvetica", 28)
        c.drawString(215, H - 245, ">")
        c.setFont("Helvetica", 12)
        c.drawString(40, H - 272, f.origin_airport.city)
        c.drawString(260, H - 272, f.dest_airport.city)

        fields = [("FLIGHT", f.number), ("DATE", f.depart.strftime("%d %b %Y")),
                  ("DEPARTS", f.depart.strftime("%H:%M")), ("ARRIVES", f.arrive.strftime("%H:%M")),
                  ("SEAT", p.seat_no), ("AIRCRAFT", f.aircraft.model)]
        x = 40
        for label, val in fields:
            c.setFont("Helvetica", 9)
            c.setFillColor(colors.grey)
            c.drawString(x, H - 335, label)
            c.setFillColor(colors.black)
            c.setFont("Helvetica-Bold", 16)
            c.drawString(x, H - 355, val)
            x += 120

        c.setDash(4, 4)
        c.line(40, H - 395, W - 40, H - 395)
        c.setDash()
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.grey)
        c.drawString(40, H - 415, "Gate closes 20 minutes before departure. Sandbox ticket generated for demonstration only.")

        key = f"{booking.ref}-{p.seat_no}".upper()
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, H - 445, f"TICKET KEY:  {key}")
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.grey)
        c.drawString(40, H - 462, "Agents scan or type this key to look the passenger up instantly.")
        widget = qr.QrCodeWidget(f"FIO|{booking.ref}|{f.number}|{p.seat_no}|{p.name}|{key}")
        x0, y0, x1, y1 = widget.getBounds()
        size = 130
        d = Drawing(size, size, transform=[size / (x1 - x0), 0, 0, size / (y1 - y0), 0, 0])
        d.add(widget)
        renderPDF.draw(d, c, W - 40 - size, H - 280)
        c.showPage()
    c.save()
    return buf.getvalue()
