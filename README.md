# Pairs Trading Model - Build Notes

**For my Behavioural Finance module**


## What's in here

This is the Python model behind my report on the LLOY/NWG pairs trade. The report is the polished academic version created for my Behavioural Finance module at my unviersity. This file is closer to build notes: how I actually put the model together, what I used to learn the parts I didn't already know, and what I'd do differently if I ran it again.

## Why I picked pairs trading

I wanted something that used the econometrics my school teaches, cointegration, OLS, hypothesis testing, but applied to something closer to a real trading problem. Pairs trading is a well-established strategy academically (Gatev et al. 2006 is the paper everyone cites), and it also forces you to build something that actually works end to end: raw prices in, trade log and P&L out. That appealed to me more than a purely theoretical exercise.

## How I built it

I didn't know how to code a cointegration test or a rolling z-score strategy before this. I picked most of it up as I went, from a few different places.

YouTube was where I started. There's a fair amount of quant finance content covering pairs trading and stat arb in Python, and a couple of videos gave me the basic shape of it before I started writing my own version, how the spread and z-score are usually set up, and roughly how entry and exit logic tends to work. I didn't lift code from any of them directly, more used them to understand the concept well enough to build it myself.

Past that, it was mostly official docs (statsmodels for the cointegration test and OLS, hmmlearn for the HMM in the Monte Carlo section), a lot of Stack Overflow for the usual debugging, pandas indexing errors, yfinance being yfinance, matplotlib doing something weird with dates. I also used Claude a fair bit while building this, mainly for debugging and for helping me understand what the statistical output actually meant, rather than just how to call a function.

The parameters and design choices in the model, the correlation band, the z-score thresholds, the transaction cost assumptions, the decision to use a rolling hedge ratio instead of a static one, are mine. I had to be able to justify each one in the report, so I couldn't just take whatever a tutorial used and leave it there.

## What the model actually does

Downloads about a year of FTSE price data, screens pairs for correlation between 0.75 and 0.95, then runs an Engle-Granger cointegration test on whatever survives that filter. For the pair that passes (LLOY/NWG), it builds a spread and a 15-day rolling z-score, using a hedge ratio that's re-estimated daily rather than fixed at the start. Trades trigger at |Z| > 1.75, exit at |Z| < 0.50, and there's a stop-loss at |Z| > 3.50. Transaction costs are built in (0.10% per leg, plus a short-borrow fee on the short side). On top of that there's a Monte Carlo simulation using a 2-state Gaussian HMM, so the strategy gets stress-tested under both calm and turbulent regimes rather than just one static assumption. Everything gets written out to Excel with charts.

## What I'd do differently

Three trades is not much of a sample, the report covers this properly, but from a purely build standpoint I'd want a longer live window or a slightly lower entry threshold next time to get more trades to actually look at. I picked the 15-day rolling window for the hedge ratio somewhat on instinct rather than testing a few alternatives, and that's probably the first thing I'd go back and do properly if I extended this.
