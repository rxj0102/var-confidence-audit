"""Which sentence of which 10-K each backtesting count comes from.

Every row names a bank, the year the count covers, the 10-K it is read from (by fiscal year),
and a phrase that occurs exactly once in that filing's text. scripts/build_disclosures.py finds
the sentence containing the phrase, checks that it states the count, and writes it with the
accession number to data/disclosures.csv. The counts below were read from those sentences.

series       primary     the one backtest each bank reports in its 10-K, at the confidence level
                         and against the revenue measure it states
             regulatory  Goldman's second series: 99% one-day regulatory VaR against positional
                         losses, which exclude intraday trading
days         trading days in the year when the filing states them, otherwise left empty
"""
from __future__ import annotations

from typing import NamedTuple


class Row(NamedTuple):
    bank: str
    year: int
    series: str
    confidence: float
    exceptions: int
    days: int | None
    filing_year: int
    phrase: str
    note: str = ""


M, R = "primary", "regulatory"

SPEC = [
    # JPMorgan. 2006 to 2009: 99% VaR of the investment bank against its market risk-related revenue.
    # From 2010: 95% firm VaR against market risk-related revenue, later called backtesting gains and losses.
    Row("JPM", 2006, M, 0.99, 0, 260, 2006, "with no loss exceeding the VAR measure"),
    Row("JPM", 2007, M, 0.99, 8, None, 2007, "losses exceeded the VAR measure on eight days"),
    Row("JPM", 2008, M, 0.99, 3, 262, 2008, "losses exceeded the VaR measure on three of those days"),
    Row("JPM", 2009, M, 0.99, 0, 261, 2009, "Losses were sustained on 42 days during the year ended December 31, 2009"),
    Row("JPM", 2010, M, 0.95, 0, 261, 2010, "losses were sustained on 13 days, none of which exceeded"),
    Row("JPM", 2011, M, 0.95, 3, None, 2011, "losses were sustained on 27 days, of which three days exceeded"),
    Row("JPM", 2012, M, 0.95, 3, 261, 2012, "the Firm sustained losses that exceeded the VaR measure on three of those days"),
    Row("JPM", 2013, M, 0.95, 2, 260, 2013, "the Firm observed two VaR band breaks"),
    Row("JPM", 2014, M, 0.95, 5, 260, 2014, "the Firm observed five VaR band breaks"),
    Row("JPM", 2015, M, 0.95, 3, 260, 2015, "the Firm observed three VaR band breaks"),
    Row("JPM", 2016, M, 0.95, 5, 260, 2016, "the Firm observed 5 VaR back-testing exceptions"),
    Row("JPM", 2017, M, 0.95, 15, 258, 2017, "the Firm observed 15 VaR back-testing exceptions"),
    Row("JPM", 2018, M, 0.95, 10, 259, 2018, "the Firm observed ten VaR back-testing exceptions"),
    Row("JPM", 2019, M, 0.95, 8, 259, 2019, "the Firm observed eight VaR back-testing exceptions"),
    Row("JPM", 2020, M, 0.95, 10, 260, 2020, "and observed 10 VaR backtesting exceptions"),
    Row("JPM", 2021, M, 0.95, 20, 260, 2021, "and observed 20 VaR backtesting exceptions"),
    Row("JPM", 2022, M, 0.95, 17, 259, 2022, "and observed 17 VaR backtesting exceptions"),
    Row("JPM", 2023, M, 0.95, 13, 258, 2023, "and observed 13 VaR backtesting exceptions"),
    Row("JPM", 2024, M, 0.95, 8, 260, 2024, "and observed eight VaR backtesting exceptions"),
    Row("JPM", 2025, M, 0.95, 12, 259, 2025, "and observed 12 VaR backtesting exceptions"),
    # Goldman Sachs. 95% VaR against daily net revenues for positions in VaR, which include intraday
    # and bid-offer revenue. Fiscal years ended in late November until 2008.
    Row("GS", 2006, M, 0.95, 3, None, 2006, "95% one-day VaR on three occasions during 2006"),
    Row("GS", 2007, M, 0.95, 10, None, 2007, "95% one-day VaR on ten occasions during 2007"),
    Row("GS", 2008, M, 0.95, 13, None, 2008, "on 13 and 10 occasions during 2008 and 2007"),
    Row("GS", 2009, M, 0.95, 0, None, 2009, "did not exceed our 95% one-day VaR during 2009"),
    Row("GS", 2010, M, 0.95, 2, None, 2010, "95% one-day VaR on two occasions during 2010"),
    Row("GS", 2011, M, 0.95, 3, None, 2011, "on three occasions during 2011 and on two occasions during 2010"),
    Row("GS", 2012, M, 0.95, 0, None, 2012, "did not exceed our 95% one-day VaR during 2012"),
    Row("GS", 2013, M, 0.95, 0, None, 2013, "did not exceed our 95% one-day VaR during 2013 or 2012"),
    Row("GS", 2014, M, 0.95, 1, None, 2014, "95% one-day VaR on one occasion during 2014 (i.e."),
    Row("GS", 2015, M, 0.95, 0, None, 2015, "did not exceed our 95% one-day VaR during 2015 (i.e."),
    Row("GS", 2016, M, 0.95, 0, None, 2016, "did not exceed our 95% one-day VaR during 2016 and 2015"),
    Row("GS", 2017, M, 0.95, 0, None, 2017, "did not exceed our 95% one-day VaR during 2017, 2016 and 2015"),
    Row("GS", 2018, M, 0.95, 2, None, 2018, "(i.e., a VaR exception) on two occasions during 2018"),
    Row("GS", 2019, M, 0.95, 0, None, 2019, "(i.e., a VaR exception) during 2019 and exceeded"),
    Row("GS", 2020, M, 0.95, 2, None, 2020, "(i.e., a VaR exception) on two occasions during 2020"),
    Row("GS", 2021, M, 0.95, 1, None, 2021, "(i.e., a VaR exception) on one occasion during 2021"),
    Row("GS", 2022, M, 0.95, 2, None, 2022, "(i.e., a VaR exception) on two occasions during 2022"),
    Row("GS", 2023, M, 0.95, 1, None, 2023, "(i.e., a VaR exception) on one occasion during 2023"),
    Row("GS", 2024, M, 0.95, 2, None, 2024, "(i.e., a VaR exception) on two occasions during 2024"),
    Row("GS", 2025, M, 0.95, 0, None, 2025, "(i.e., a VaR exception) during 2025"),
    # Goldman Sachs, 99% regulatory VaR against positional losses.
    Row("GS", 2015, R, 0.99, 0, None, 2016, "did not exceed its 99% one-day regulatory VaR during 2015"),
    Row("GS", 2016, R, 0.99, 2, None, 2016, "regulatory VaR on two occasions during 2016 and did not"),
    Row("GS", 2017, R, 0.99, 0, None, 2017, "did not exceed its 99% one-day regulatory VaR during 2017 and exceeded"),
    Row("GS", 2018, R, 0.99, 2, None, 2018, "regulatory VaR on two occasions during 2018 and did not"),
    Row("GS", 2019, R, 0.99, 1, None, 2019, "regulatory VaR on one occasion during 2019"),
    Row("GS", 2020, R, 0.99, 6, None, 2020, "regulatory VaR on six occasions during 2020"),
    Row("GS", 2021, R, 0.99, 1, None, 2021, "regulatory VaR on one occasion during 2021 and on six"),
    Row("GS", 2022, R, 0.99, 1, None, 2022, "on one occasion during each of the years ended 2022 and 2021"),
    Row("GS", 2023, R, 0.99, 1, None, 2023, "on one occasion during both 2023 and 2022"),
    Row("GS", 2024, R, 0.99, 2, None, 2024, "regulatory VaR on two occasions during 2024"),
    Row("GS", 2025, R, 0.99, 3, None, 2025, "regulatory VaR on three occasions during 2025"),
    # Morgan Stanley. 95% Trading VaR, later Total Management VaR, against daily net trading revenue,
    # which includes intraday trading and excludes fees, commissions and net interest income.
    # Fiscal years ended November 30 until 2008.
    Row("MS", 2006, M, 0.95, 1, None, 2006, "net trading loss of $87 million on one business day during fiscal 2006"),
    Row("MS", 2007, M, 0.95, 15, None, 2007, "Trading VaR on 15 days during fiscal 2007"),
    Row("MS", 2008, M, 0.95, 18, None, 2008, "Trading VaR on 18 days during fiscal 2008"),
    Row("MS", 2009, M, 0.95, 1, None, 2009, "Trading VaR on one day during 2009"),
    Row("MS", 2010, M, 0.95, 0, None, 2010, "with zero excesses of the 95%/one-day Trading VaR"),
    Row("MS", 2011, M, 0.95, 4, None, 2011, "4 days of which were in excess of the 95%/one-day Trading VaR"),
    Row("MS", 2012, M, 0.95, 0, None, 2012, "net trading losses on 37 days, of which no day"),
    Row("MS", 2013, M, 0.95, 1, None, 2013, "of which 1 day was in excess of the 95%/one-day Management VaR"),
    Row("MS", 2014, M, 0.95, 0, None, 2014, "net trading losses on 32 days, of which no day was in excess of the 95%/one-day Total"),
    Row("MS", 2015, M, 0.95, 0, None, 2015, "During 2015, the Company experienced net trading losses on 32 days"),
    Row("MS", 2016, M, 0.95, 0, None, 2016, "net trading losses on 18 days, of which no day"),
    Row("MS", 2017, M, 0.95, 0, None, 2017, "net trading losses on 15 days, which were not in excess"),
    Row("MS", 2018, M, 0.95, 0, None, 2018, "There were no days in 2018 on which trading losses exceeded VaR"),
    Row("MS", 2019, M, 0.95, 0, None, 2019, "There were no days in 2019 on which trading losses exceeded VaR"),
    Row("MS", 2020, M, 0.95, 1, None, 2020, "seven trading loss days in 2020, one of which exceeded"),
    Row("MS", 2021, M, 0.95, 1, None, 2021, "There were 14 trading loss days in 2021, one of which exceeded"),
    Row("MS", 2022, M, 0.95, 0, None, 2022, "There were 15 trading loss days in 2022, none of which exceeded"),
    Row("MS", 2023, M, 0.95, 1, None, 2023, "There were 16 trading loss days in 2023, one of which exceeded"),
    Row("MS", 2024, M, 0.95, 0, None, 2024, "There were 11 trading loss days in 2024, none of which exceeded"),
    Row("MS", 2025, M, 0.95, 0, None, 2025, "There were six trading loss days in 2025, none of which exceeded"),
    # Bank of America. 99% VaR of the total market-based (later covered) portfolio against
    # trading-related revenue (from 2020, the subset of it that VaR covers).
    Row("BAC", 2006, M, 0.99, 0, None, 2007, "losses did not exceed daily trading VAR in the twelve months ended December 31, 2006"),
    Row("BAC", 2007, M, 0.99, 14, None, 2007, "actual losses exceeded daily trading VAR fourteen times",
        "excludes discrete CDO writedowns, so the true count is higher"),
    Row("BAC", 2008, M, 0.99, 2, None, 2008, "Actual losses exceeded daily trading VAR two times in the 12 months ended December 31, 2008"),
    Row("BAC", 2009, M, 0.99, 0, None, 2009, "did not exceed daily trading VAR in the twelve months ended December 31, 2009"),
    Row("BAC", 2010, M, 0.99, 0, None, 2010, "did not exceed daily trading VaR in the twelve months ended December 31, 2010"),
    Row("BAC", 2011, M, 0.99, 0, None, 2011, "did not exceed daily trading VaR in 2011 or 2010"),
    Row("BAC", 2012, M, 0.99, 0, None, 2012, "did not exceed daily trading VaR in 2012 or 2011"),
    Row("BAC", 2013, M, 0.99, 0, None, 2013, "There were no days with backtesting excesses"),
    Row("BAC", 2014, M, 0.99, 0, None, 2014, "During 2014 , there were no days in which there was a backtesting excess"),
    Row("BAC", 2015, M, 0.99, 0, None, 2015, "During 2015 , there were no days in which there was a backtesting excess"),
    Row("BAC", 2016, M, 0.99, 0, None, 2016, "During 2016 , there were no days in which there was a backtesting excess"),
    Row("BAC", 2017, M, 0.99, 0, None, 2017, "During 2017 , there were no days in which there was a backtesting excess"),
    Row("BAC", 2018, M, 0.99, 3, None, 2018, "During 2018 , there were three days in which there was a backtesting excess"),
    Row("BAC", 2019, M, 0.99, 0, None, 2019, "During 2019 , there were no days in which there was a backtesting excess"),
    Row("BAC", 2020, M, 0.99, 7, None, 2020, "During 2020, there were seven days"),
    Row("BAC", 2021, M, 0.99, 2, None, 2021, "During 2021, there were two days"),
    Row("BAC", 2022, M, 0.99, 1, None, 2022, "During 2022, there was one day"),
    Row("BAC", 2023, M, 0.99, 0, None, 2023, "During 2023, there were no days"),
    Row("BAC", 2024, M, 0.99, 1, None, 2024, "During 2024, there was one day"),
    Row("BAC", 2025, M, 0.99, 0, None, 2025, "During 2025, there were no days"),
    # Citigroup. 99% VaR; total trading VaR in 2011 and 2012, Regulatory VaR on covered positions
    # against buy-and-hold P&L from 2013. The 2006 to 2010 10-Ks report no count.
    Row("C", 2011, M, 0.99, 1, None, 2011, "During the last four quarters, there was one back-testing exception"),
    Row("C", 2012, M, 0.99, 0, None, 2012, "In 2012, no back-testing exceptions were observed"),
    Row("C", 2013, M, 0.99, 0, None, 2013, "there were no back-testing exceptions observed for Citi’s Regulatory VAR"),
    Row("C", 2014, M, 0.99, 1, None, 2014, "there was one back testing exception where trading losses exceeded"),
    Row("C", 2015, M, 0.99, 1, None, 2015, "As of December 31, 2015, there was one back-testing exception"),
    Row("C", 2016, M, 0.99, 2, None, 2016, "During 2016, there were two back-testing exceptions"),
    Row("C", 2017, M, 0.99, 0, None, 2017, "During 2017, there were no back-testing exceptions"),
    Row("C", 2018, M, 0.99, 1, None, 2018, "As of December 31, 2018, there was one back-testing exception"),
    Row("C", 2019, M, 0.99, 0, None, 2019, "As of December 31, 2019, there were no back-testing exceptions"),
    Row("C", 2020, M, 0.99, 4, None, 2020, "As of December 31, 2020, four back-testing exceptions"),
    Row("C", 2021, M, 0.99, 1, None, 2021, "As of December 31, 2021, one back-testing exception"),
    Row("C", 2022, M, 0.99, 1, None, 2022, "During 2022, one back-testing exception"),
    Row("C", 2023, M, 0.99, 3, None, 2023, "During 2023, three back-testing exceptions"),
    Row("C", 2024, M, 0.99, 0, None, 2024, "During 2024, no back-testing exceptions"),
    Row("C", 2025, M, 0.99, 0, None, 2025, "During 2025, no back-testing exceptions"),
]

WORDS = {"no": 0, "none": 0, "zero": 0, "not": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
         "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
         "thirteen": 13, "fourteen": 14, "fifteen": 15}


def states_count(sentence: str, n: int) -> bool:
    """True if the sentence states the number n as digits or a word (zero as no, none or not)."""
    import re
    if re.search(rf"(?<![\d.,$]){n}(?!\d)(?![.,]\d)", sentence):
        return True
    return any(v == n and re.search(rf"(?i)\b{w}\b", sentence) for w, v in WORDS.items())
