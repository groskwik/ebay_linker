@echo off
    
python ebay_scrape.py --headless --stdout-short

python restock.py 

python ebay_linker.py ^
  --orders-csv awaiting_shipment_items.csv ^
  --links-json ebay_links.json ^
  --out-links-json ebay_links.json ^
  --recursive ^
  --min-score 60 ^
  --min-margin 8 ^
  --print 

rem  --always-ask-printer ^
rem --printer 2 ^
