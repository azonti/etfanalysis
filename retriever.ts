import puppeteer from 'puppeteer';
import fs from 'node:fs/promises';

interface Config {
  readonly name: string;
  readonly tickerSymbol: string;
  readonly pathToJSON: string;
}

const configs: readonly Config[] = [
  {
    name: 'SPX_EST_USD',
    tickerSymbol: '^GSPC',
    pathToJSON: './data/SPX_EST_USD/SPX_EST_USD.json',
  },
  {
    name: 'SPY_EST_USD',
    tickerSymbol: 'SPY',
    pathToJSON: './data/SPX_EST_USD/SPY_EST_USD.json',
  },
  {
    name: 'SSO_EST_USD',
    tickerSymbol: 'SSO',
    pathToJSON: './data/SPX_EST_USD/SSO_EST_USD.json',
  },
  {
    name: 'SPXL_EST_USD',
    tickerSymbol: 'SPXL',
    pathToJSON: './data/SPX_EST_USD/SPXL_EST_USD.json',
  },
  {
    name: 'SPXS_EST_USD',
    tickerSymbol: 'SPXS',
    pathToJSON: './data/SPX_EST_USD/SPXS_EST_USD.json',
  },
  {
    name: 'NDX_EST_USD',
    tickerSymbol: '^NDX',
    pathToJSON: './data/NDX_EST_USD/NDX_EST_USD.json',
  },
  {
    name: 'QQQ_EST_USD',
    tickerSymbol: 'QQQ',
    pathToJSON: './data/NDX_EST_USD/QQQ_EST_USD.json',
  },
  {
    name: 'QLD_EST_USD',
    tickerSymbol: 'QLD',
    pathToJSON: './data/NDX_EST_USD/QLD_EST_USD.json',
  },
  {
    name: 'TQQQ_EST_USD',
    tickerSymbol: 'TQQQ',
    pathToJSON: './data/NDX_EST_USD/TQQQ_EST_USD.json',
  },
  {
    name: 'SQQQ_EST_USD',
    tickerSymbol: 'SQQQ',
    pathToJSON: './data/NDX_EST_USD/SQQQ_EST_USD.json',
  },
  {
    name: 'SOX_EST_USD',
    tickerSymbol: '^SOX',
    pathToJSON: './data/SOX_EST_USD/SOX_EST_USD.json',
  },
  {
    name: 'SOXX_EST_USD',
    tickerSymbol: 'SOXX',
    pathToJSON: './data/SOX_EST_USD/SOXX_EST_USD.json',
  },
  {
    name: 'SOXL_EST_USD',
    tickerSymbol: 'SOXL',
    pathToJSON: './data/SOX_EST_USD/SOXL_EST_USD.json',
  },
  {
    name: 'SOXS_EST_USD',
    tickerSymbol: 'SOXS',
    pathToJSON: './data/SOX_EST_USD/SOXS_EST_USD.json',
  },
  {
    name: 'XLK_EST_USD',
    tickerSymbol: 'XLK',
    pathToJSON: './data/IXT_EST_USD/XLK_EST_USD.json',
  },
  {
    name: 'ROM_EST_USD',
    tickerSymbol: 'ROM',
    pathToJSON: './data/IXT_EST_USD/ROM_EST_USD.json',
  },
  {
    name: 'TECL_EST_USD',
    tickerSymbol: 'TECL',
    pathToJSON: './data/IXT_EST_USD/TECL_EST_USD.json',
  },
  {
    name: 'TECS_EST_USD',
    tickerSymbol: 'TECS',
    pathToJSON: './data/IXT_EST_USD/TECS_EST_USD.json',
  },
  {
    name: 'VXUS_EST_USD',
    tickerSymbol: 'VXUS',
    pathToJSON: './data/ACXUSS_EST_USD/VXUS_EST_USD.json',
  },
  {
    name: 'VEU_EST_USD',
    tickerSymbol: 'VEU',
    pathToJSON: './data/AW02_EST_USD/VEU_EST_USD.json',
  },
  {
    name: '1322.T_JST_JPY',
    tickerSymbol: '1322.T',
    pathToJSON: './data/000300.SS_JST_JPY/1322.T_JST_JPY.json',
  },
  {
    name: 'ASHR_EST_USD',
    tickerSymbol: 'ASHR',
    pathToJSON: './data/000300.SS_EST_USD/ASHR_EST_USD.json',
  },
  {
    name: 'CHAU_EST_USD',
    tickerSymbol: 'CHAU',
    pathToJSON: './data/000300.SS_EST_USD/CHAU_EST_USD.json',
  },
  {
    name: 'DAX_CET_EUR',
    tickerSymbol: '^GDAXI',
    pathToJSON: './data/DAX_CET_EUR/DAX_CET_EUR.json',
  },
  {
    name: 'EXS1.DE_CET_EUR',
    tickerSymbol: 'EXS1.DE',
    pathToJSON: './data/DAX_CET_EUR/EXS1.DE_CET_EUR.json',
  },
  {
    name: 'LYY8.DE_CET_EUR',
    tickerSymbol: 'LYY8.DE',
    pathToJSON: './data/DAX_CET_EUR/LYY8.DE_CET_EUR.json',
  },
  {
    name: 'SX5E_CET_EUR',
    tickerSymbol: '^STOXX50E',
    pathToJSON: './data/SX5E_CET_EUR/SX5E_CET_EUR.json',
  },
  {
    name: 'EXW1.DE_CET_EUR',
    tickerSymbol: 'EXW1.DE',
    pathToJSON: './data/SX5E_CET_EUR/EXW1.DE_CET_EUR.json',
  },
  {
    name: 'LYMZ.DE_CET_EUR',
    tickerSymbol: 'LYMZ.DE',
    pathToJSON: './data/SX5E_CET_EUR/LYMZ.DE_CET_EUR.json',
  },
  {
    name: 'N225_JST_JPY',
    tickerSymbol: '^N225',
    pathToJSON: './data/N225_JST_JPY/N225_JST_JPY.json',
  },
  {
    name: '1321.T_JST_JPY',
    tickerSymbol: '1321.T',
    pathToJSON: './data/N225_JST_JPY/1321.T_JST_JPY.json',
  },
  {
    name: '1579.T_JST_JPY',
    tickerSymbol: '1579.T',
    pathToJSON: './data/N225_JST_JPY/1579.T_JST_JPY.json',
  },
  {
    name: '1306.T_JST_JPY',
    tickerSymbol: '1306.T',
    pathToJSON: './data/TOPIX_JST_JPY/1306.T_JST_JPY.json',
  },
  {
    name: '1568.T_JST_JPY',
    tickerSymbol: '1568.T',
    pathToJSON: './data/TOPIX_JST_JPY/1568.T_JST_JPY.json',
  },
  {
    name: 'ISFA.AS_CET_EUR',
    tickerSymbol: 'ISFA.AS',
    pathToJSON: './data/UKX_CET_EUR/ISFA.AS_CET_EUR.json',
  },
  {
    name: 'ISF.L_GMT_GBP',
    tickerSymbol: 'ISF.L',
    pathToJSON: './data/UKX_GMT_GBP/ISF.L_GMT_GBP.json',
  },
  {
    name: 'LUK2.L_GMT_GBP',
    tickerSymbol: 'LUK2.L',
    pathToJSON: './data/UKX_GMT_GBP/LUK2.L_GMT_GBP.json',
  },
  {
    name: 'EWY_EST_USD',
    tickerSymbol: 'EWY',
    pathToJSON: './data/M1KR2550_EST_USD/EWY_EST_USD.json',
  },
  {
    name: 'EWT_EST_USD',
    tickerSymbol: 'EWT',
    pathToJSON: './data/M1CXBISD_EST_USD/EWT_EST_USD.json',
  },
];

async function delay(ms: number): Promise<void> {
  await new Promise(resolve => setTimeout(resolve, ms));
}

async function retry(fn: () => Promise<void>, nRetries = 5, delayMs = 1000): Promise<void> {
  for (let i = 0; i < nRetries; i++) {
    try {
      await fn();
      return;
    } catch (error) {
      console.warn(`Attempt ${String(i + 1)} failed:`, error);
      if (i + 1 === nRetries) {
        throw new Error(`Failed after ${String(nRetries)} attempts`, { cause: error });
      }
      await delay(delayMs);
    }
  }
}

const browser = await puppeteer.launch();
const page = await browser.newPage();

for (const config of configs) {
  console.debug('Navigating to Yahoo Finance for:', config.name);
  await page.goto(`https://finance.yahoo.com/quote/${encodeURIComponent(config.tickerSymbol)}/history/`);

  console.debug('Interacting with date picker for:', config.name);
  await retry(async () => {
    await page.click('[data-testid="history-date-picker"] > button');
    await delay(1000);
    await page.click('[data-testid="history-date-picker"] button[value="MAX"]');
  });
  await delay(30000);

  console.debug('Extracting data from the page for:', config.name);
  const data = await page.$eval(
    '[data-testid="history-table"] tbody',
    tbody => Array.from(tbody.children).filter(
      tr => tr.children.length === 7
    ).map(
      tr => Array.from(tr.children).map(
        td => td.innerHTML.trim()
      )
    ),
  );

  console.debug('Writing data to file for:', config.name);
  await fs.mkdir(config.pathToJSON.substring(0, config.pathToJSON.lastIndexOf('/')), { recursive: true });
  await fs.writeFile(config.pathToJSON, JSON.stringify(data));
}

await page.close();
await browser.close();
