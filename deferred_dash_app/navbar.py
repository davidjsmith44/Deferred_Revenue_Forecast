"""# navbar.py"""
import dash_bootstrap_components as dbc


def Navbar():
    navbar = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Time-Series", href="/time-series")),
            dbc.DropdownMenu(
                nav=True,
                in_navbar=True,
                label="Menu",
                children=[
                    dbc.DropdownMenuItem("Document Currency Billings"),
                    dbc.DropdownMenuItem("US Eqivalent of Billings"),
                    dbc.DropdownMenuItem(divider=True),
                    dbc.DropdownMenuItem("Deferred Revenue"),
                ],
            ),
        ],
        brand="Home",
        brand_href="/home",
        sticky="top",
    )

    return navbar
