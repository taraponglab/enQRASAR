import pandas as pd
import enraqsar as enq
import os
import typer
from rich import print
from rich.console import Console
from rich.table import Table

def main(input: str = typer.Option("example.xlsx", help="Input file name for example example.xlsx"), 
         output: str = typer.Option("output.csv", help="Output file name for example output.csv")):
    typer.secho('This software calculate the pIC50 of keratinocyte cell viability from Chemical SMILES', fg=typer.colors.WHITE, bold=True)
    typer.secho('Please wait for a while', fg=typer.colors.WHITE, bold=True)

    test   = pd.read_excel(os.path.join('input',input), index_col='LigandID')
    result = enq.enraqsar(test)
    result.to_csv(os.path.join('output', output))
    

    
    console = Console()
    table = Table(show_header=True, header_style="bold green")
    table.add_column("LigandID")
    for column in result.columns:
        table.add_column(column)

    for index, row in result.iterrows():
        table.add_row(index, *row.round(2).astype(str).tolist())

    console.print(table)

    typer.secho('The result has been saved in output folder', fg=typer.colors.WHITE, bold=True)
    typer.secho('Thank you for using this software', fg=typer.colors.WHITE, bold=True)

if __name__ == '__main__':
    typer.run(main)