import crypto from 'node:crypto';

export interface Config {
  readonly name: string;
  readonly pathToJSON: string;
  readonly pathToSupplementaryJSON?: string;
  readonly supplementaryLeverage?: number;
}

export const configs: readonly Config[] = [
  {
    name: 'SPX',
    pathToJSON: './data/SPX_EST_USD/SPY_EST_USD.json',
    pathToSupplementaryJSON: './data/SPX_EST_USD/SPX_EST_USD.json',
  },
  {
    name: '2 SPX',
    pathToJSON: './data/SPX_EST_USD/SSO_EST_USD.json',
    pathToSupplementaryJSON: './data/SPX_EST_USD/SPX_EST_USD.json',
    supplementaryLeverage: 2,
  },
  {
    name: '3 SPX',
    pathToJSON: './data/SPX_EST_USD/SPXL_EST_USD.json',
    pathToSupplementaryJSON: './data/SPX_EST_USD/SPX_EST_USD.json',
    supplementaryLeverage: 3,
  },
  {
    name: '-3 SPX',
    pathToJSON: './data/SPX_EST_USD/SPXS_EST_USD.json',
    pathToSupplementaryJSON: './data/SPX_EST_USD/SPX_EST_USD.json',
    supplementaryLeverage: -3,
  },
  {
    name: 'NDX',
    pathToJSON: './data/NDX_EST_USD/QQQ_EST_USD.json',
    pathToSupplementaryJSON: './data/NDX_EST_USD/NDX_EST_USD.json',
  },
  {
    name: '2 NDX',
    pathToJSON: './data/NDX_EST_USD/QLD_EST_USD.json',
    pathToSupplementaryJSON: './data/NDX_EST_USD/NDX_EST_USD.json',
    supplementaryLeverage: 2,
  },
  {
    name: '3 NDX',
    pathToJSON: './data/NDX_EST_USD/TQQQ_EST_USD.json',
    pathToSupplementaryJSON: './data/NDX_EST_USD/NDX_EST_USD.json',
    supplementaryLeverage: 3,
  },
  {
    name: '-3 NDX',
    pathToJSON: './data/NDX_EST_USD/SQQQ_EST_USD.json',
    pathToSupplementaryJSON: './data/NDX_EST_USD/NDX_EST_USD.json',
    supplementaryLeverage: -3,
  },
  {
    name: 'SOX',
    pathToJSON: './data/SOX_EST_USD/SOXX_EST_USD.json',
    pathToSupplementaryJSON: './data/SOX_EST_USD/SOX_EST_USD.json',
  },
  {
    name: '3 SOX',
    pathToJSON: './data/SOX_EST_USD/SOXL_EST_USD.json',
    pathToSupplementaryJSON: './data/SOX_EST_USD/SOX_EST_USD.json',
    supplementaryLeverage: 3,
  },
  {
    name: '-3 SOX',
    pathToJSON: './data/SOX_EST_USD/SOXS_EST_USD.json',
    pathToSupplementaryJSON: './data/SOX_EST_USD/SOX_EST_USD.json',
    supplementaryLeverage: -3,
  },
  {
    name: 'IXT',
    pathToJSON: './data/IXT_EST_USD/XLK_EST_USD.json',
  },
  {
    name: '2 IXT',
    pathToJSON: './data/IXT_EST_USD/ROM_EST_USD.json',
    pathToSupplementaryJSON: './data/IXT_EST_USD/XLK_EST_USD.json',
    supplementaryLeverage: 2,
  },
  {
    name: '3 IXT',
    pathToJSON: './data/IXT_EST_USD/TECL_EST_USD.json',
    pathToSupplementaryJSON: './data/IXT_EST_USD/XLK_EST_USD.json',
    supplementaryLeverage: 3,
  },
  {
    name: '-3 IXT',
    pathToJSON: './data/IXT_EST_USD/TECS_EST_USD.json',
    pathToSupplementaryJSON: './data/IXT_EST_USD/XLK_EST_USD.json',
    supplementaryLeverage: -3,
  },
  {
    name: 'ACXUSS',
    pathToJSON: './data/ACXUSS_EST_USD/VXUS_EST_USD.json',
  },
  {
    name: 'AW02',
    pathToJSON: './data/AW02_EST_USD/VEU_EST_USD.json',
  },
  {
    name: '000300.SS (JPY)',
    pathToJSON: './data/000300.SS_JST_JPY/1322.T_JST_JPY.json',
  },
  {
    name: '000300.SS (USD)',
    pathToJSON: './data/000300.SS_EST_USD/ASHR_EST_USD.json',
  },
  {
    name: '2 000300.SS (USD)',
    pathToJSON: './data/000300.SS_EST_USD/CHAU_EST_USD.json',
    pathToSupplementaryJSON: './data/000300.SS_EST_USD/ASHR_EST_USD.json',
    supplementaryLeverage: 2,
  },
  {
    name: 'DAX',
    pathToJSON: './data/DAX_CET_EUR/EXS1.DE_CET_EUR.json',
    pathToSupplementaryJSON: './data/DAX_CET_EUR/DAX_CET_EUR.json',
  },
  {
    name: '2 DAX',
    pathToJSON: './data/DAX_CET_EUR/LYY8.DE_CET_EUR.json',
    pathToSupplementaryJSON: './data/DAX_CET_EUR/DAX_CET_EUR.json',
    supplementaryLeverage: 2,
  },
  {
    name: 'SX5E',
    pathToJSON: './data/SX5E_CET_EUR/EXW1.DE_CET_EUR.json',
    pathToSupplementaryJSON: './data/SX5E_CET_EUR/SX5E_CET_EUR.json',
  },
  {
    name: '2 SX5E',
    pathToJSON: './data/SX5E_CET_EUR/LYMZ.DE_CET_EUR.json',
    pathToSupplementaryJSON: './data/SX5E_CET_EUR/SX5E_CET_EUR.json',
    supplementaryLeverage: 2,
  },
  {
    name: 'N225',
    pathToJSON: './data/N225_JST_JPY/1321.T_JST_JPY.json',
    pathToSupplementaryJSON: './data/N225_JST_JPY/N225_JST_JPY.json',
  },
  {
    name: '2 N225',
    pathToJSON: './data/N225_JST_JPY/1579.T_JST_JPY.json',
    pathToSupplementaryJSON: './data/N225_JST_JPY/N225_JST_JPY.json',
    supplementaryLeverage: 2,
  },
  {
    name: 'TOPIX',
    pathToJSON: './data/TOPIX_JST_JPY/1306.T_JST_JPY.json',
  },
  {
    name: '2 TOPIX',
    pathToJSON: './data/TOPIX_JST_JPY/1568.T_JST_JPY.json',
    pathToSupplementaryJSON: './data/TOPIX_JST_JPY/1306.T_JST_JPY.json',
    supplementaryLeverage: 2,
  },
  {
    name: 'UKX (EUR)',
    pathToJSON: './data/UKX_CET_EUR/ISFA.AS_CET_EUR.json',
  },
  {
    name: 'UKX',
    pathToJSON: './data/UKX_GMT_GBP/ISF.L_GMT_GBP.json',
  },
  {
    name: '2 UKX',
    pathToJSON: './data/UKX_GMT_GBP/LUK2.L_GMT_GBP.json',
    pathToSupplementaryJSON: './data/UKX_GMT_GBP/ISF.L_GMT_GBP.json',
    supplementaryLeverage: 2,
  },
  {
    name: 'M1KR2550 (USD)',
    pathToJSON: './data/M1KR2550_EST_USD/EWY_EST_USD.json',
  },
  {
    name: 'M1CXBISD (USD)',
    pathToJSON: './data/M1CXBISD_EST_USD/EWT_EST_USD.json',
  },
];



const md5HashCache = new Map<string, string>();
export function md5Hash(data: string): string {
  const cachedHash = md5HashCache.get(data);
  if (cachedHash) {
    return cachedHash;
  }
  const hash = crypto.createHash('md5').update(data).digest('hex');
  md5HashCache.set(data, hash);
  return hash;
}
