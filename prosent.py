antall_besøkende = 4150                                     # Oppgir hvor mange besøkende det er i 2020
årlig_vekst = 1.05                                          # Oppgir vekstfaktoren til prosentøkninga som desimaltall

while antall_besøkende < 5300:                              # Innleder ei løkke som skal repetere innholdet sitt så lenge antall besøkende er mindre enn 5300
    print(antall_besøkende)                                 # Skriver ut hvor mange besøkende det er hvert år.
    antall_besøkende = antall_besøkende * årlig_vekst       # Her oppdaterer jeg verdien til antall besøkende. Den blir satt til å øke med 5% hver gang innholdet i løkka kjører.

