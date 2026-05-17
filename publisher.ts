import util from 'node:util';
import { execFile } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import { createWriteStream } from 'node:fs';

interface Config {
  readonly name: string;
  readonly pathToJSON: string;
  readonly pathToSupplementaryJSON?: string;
  readonly supplementaryLeverage?: number;
}

const configs: readonly Config[] = [
  {
    name: 'SPX',
    pathToJSON: './SPX_EST_USD/SPY_EST_USD.json',
    pathToSupplementaryJSON: './SPX_EST_USD/SPX_EST_USD.json',
  },
  {
    name: '2 SPX',
    pathToJSON: './SPX_EST_USD/SSO_EST_USD.json',
    pathToSupplementaryJSON: './SPX_EST_USD/SPX_EST_USD.json',
    supplementaryLeverage: 2,
  },
  {
    name: '3 SPX',
    pathToJSON: './SPX_EST_USD/SPXL_EST_USD.json',
    pathToSupplementaryJSON: './SPX_EST_USD/SPX_EST_USD.json',
    supplementaryLeverage: 3,
  },
  {
    name: '-3 SPX',
    pathToJSON: './SPX_EST_USD/SPXS_EST_USD.json',
    pathToSupplementaryJSON: './SPX_EST_USD/SPX_EST_USD.json',
    supplementaryLeverage: -3,
  },
  {
    name: 'NDX',
    pathToJSON: './NDX_EST_USD/QQQ_EST_USD.json',
    pathToSupplementaryJSON: './NDX_EST_USD/NDX_EST_USD.json',
  },
  {
    name: '2 NDX',
    pathToJSON: './NDX_EST_USD/QLD_EST_USD.json',
    pathToSupplementaryJSON: './NDX_EST_USD/NDX_EST_USD.json',
    supplementaryLeverage: 2,
  },
  {
    name: '3 NDX',
    pathToJSON: './NDX_EST_USD/TQQQ_EST_USD.json',
    pathToSupplementaryJSON: './NDX_EST_USD/NDX_EST_USD.json',
    supplementaryLeverage: 3,
  },
  {
    name: '-3 NDX',
    pathToJSON: './NDX_EST_USD/SQQQ_EST_USD.json',
    pathToSupplementaryJSON: './NDX_EST_USD/NDX_EST_USD.json',
    supplementaryLeverage: -3,
  },
  {
    name: 'SOX',
    pathToJSON: './SOX_EST_USD/SOXX_EST_USD.json',
    pathToSupplementaryJSON: './SOX_EST_USD/SOX_EST_USD.json',
  },
  {
    name: '3 SOX',
    pathToJSON: './SOX_EST_USD/SOXL_EST_USD.json',
    pathToSupplementaryJSON: './SOX_EST_USD/SOX_EST_USD.json',
    supplementaryLeverage: 3,
  },
  {
    name: '-3 SOX',
    pathToJSON: './SOX_EST_USD/SOXS_EST_USD.json',
    pathToSupplementaryJSON: './SOX_EST_USD/SOX_EST_USD.json',
    supplementaryLeverage: -3,
  },
  {
    name: 'IXT',
    pathToJSON: './IXT_EST_USD/XLK_EST_USD.json',
  },
  {
    name: '2 IXT',
    pathToJSON: './IXT_EST_USD/ROM_EST_USD.json',
    pathToSupplementaryJSON: './IXT_EST_USD/XLK_EST_USD.json',
    supplementaryLeverage: 2,
  },
  {
    name: '3 IXT',
    pathToJSON: './IXT_EST_USD/TECL_EST_USD.json',
    pathToSupplementaryJSON: './IXT_EST_USD/XLK_EST_USD.json',
    supplementaryLeverage: 3,
  },
  {
    name: '-3 IXT',
    pathToJSON: './IXT_EST_USD/TECS_EST_USD.json',
    pathToSupplementaryJSON: './IXT_EST_USD/XLK_EST_USD.json',
    supplementaryLeverage: -3,
  },
  {
    name: 'ACXUSS',
    pathToJSON: './ACXUSS_EST_USD/VXUS_EST_USD.json',
  },
  {
    name: 'AW02',
    pathToJSON: './AW02_EST_USD/VEU_EST_USD.json',
  },
  {
    name: '000300.SS (JPY)',
    pathToJSON: './000300.SS_JST_JPY/1322.T_JST_JPY.json',
  },
  {
    name: '000300.SS (USD)',
    pathToJSON: './000300.SS_EST_USD/ASHR_EST_USD.json',
  },
  {
    name: '2 000300.SS (USD)',
    pathToJSON: './000300.SS_EST_USD/CHAU_EST_USD.json',
    pathToSupplementaryJSON: './000300.SS_EST_USD/ASHR_EST_USD.json',
    supplementaryLeverage: 2,
  },
  {
    name: 'DAX',
    pathToJSON: './DAX_CET_EUR/EXS1.DE_CET_EUR.json',
    pathToSupplementaryJSON: './DAX_CET_EUR/DAX_CET_EUR.json',
  },
  {
    name: '2 DAX',
    pathToJSON: './DAX_CET_EUR/LYY8.DE_CET_EUR.json',
    pathToSupplementaryJSON: './DAX_CET_EUR/DAX_CET_EUR.json',
    supplementaryLeverage: 2,
  },
  {
    name: 'SX5E',
    pathToJSON: './SX5E_CET_EUR/EXW1.DE_CET_EUR.json',
    pathToSupplementaryJSON: './SX5E_CET_EUR/SX5E_CET_EUR.json',
  },
  {
    name: '2 SX5E',
    pathToJSON: './SX5E_CET_EUR/LYMZ.DE_CET_EUR.json',
    pathToSupplementaryJSON: './SX5E_CET_EUR/SX5E_CET_EUR.json',
    supplementaryLeverage: 2,
  },
  {
    name: 'N225',
    pathToJSON: './N225_JST_JPY/1321.T_JST_JPY.json',
    pathToSupplementaryJSON: './N225_JST_JPY/N225_JST_JPY.json',
  },
  {
    name: '2 N225',
    pathToJSON: './N225_JST_JPY/1579.T_JST_JPY.json',
    pathToSupplementaryJSON: './N225_JST_JPY/N225_JST_JPY.json',
    supplementaryLeverage: 2,
  },
  {
    name: 'TOPIX',
    pathToJSON: './TOPIX_JST_JPY/1306.T_JST_JPY.json',
  },
  {
    name: '2 TOPIX',
    pathToJSON: './TOPIX_JST_JPY/1568.T_JST_JPY.json',
    pathToSupplementaryJSON: './TOPIX_JST_JPY/1306.T_JST_JPY.json',
    supplementaryLeverage: 2,
  },
  {
    name: 'UKX (EUR)',
    pathToJSON: './UKX_CET_EUR/ISFA.AS_CET_EUR.json',
  },
  {
    name: 'UKX',
    pathToJSON: './UKX_GMT_GBP/ISF.L_GMT_GBP.json',
  },
  {
    name: '2 UKX',
    pathToJSON: './UKX_GMT_GBP/LUK2.L_GMT_GBP.json',
    pathToSupplementaryJSON: './UKX_GMT_GBP/ISF.L_GMT_GBP.json',
    supplementaryLeverage: 2,
  },
  {
    name: 'M1KR2550 (USD)',
    pathToJSON: './M1KR2550_EST_USD/EWY_EST_USD.json',
  },
  {
    name: 'M1CXBISD (USD)',
    pathToJSON: './M1CXBISD_EST_USD/EWT_EST_USD.json',
  },
];

const md5HashCache = new Map<string, string>();
function md5Hash(data: string): string {
  const cachedHash = md5HashCache.get(data);
  if (cachedHash) {
    return cachedHash;
  }
  const hash = crypto.createHash('md5').update(data).digest('hex');
  md5HashCache.set(data, hash);
  return hash;
}

interface AnnualReturnPyOutput {
  leverage: number;
  supplementaryLeverage?: number;
  hasAbnormalDailyReturn?: boolean;
  lastDate: string;
  firstDate: string;
  lowerBoundOfAnnualVolatility: number;
  upperBoundOfAnnualVolatility: number;
  lowerBoundOfAnnualDriftMinusAnnualVolatilitySquaredHalved: number;
  upperBoundOfAnnualDriftMinusAnnualVolatilitySquaredHalved: number;
  sublowerBoundOfAnnualDriftMinusAnnualVolatilitySquaredHalved: number;
  subupperBoundOfAnnualDriftMinusAnnualVolatilitySquaredHalved: number;
  lowerBoundOfMedianOfAnnualReturn: number;
  upperBoundOfMedianOfAnnualReturn: number;
  sublowerBoundOfMedianOfAnnualReturn: number;
  subupperBoundOfMedianOfAnnualReturn: number;
}

function annualReturnPyStdoutParser(stdout: string): AnnualReturnPyOutput {
  const lines = stdout.trim().split('\n');
  const output: Partial<AnnualReturnPyOutput> = {};
  for (const line of lines) {
    switch (true) {
      case !line.trim():
        break;
      case line.startsWith('Leverage:'):
        output.leverage = parseFloat(line.replace('Leverage:', '').trim());
        break;
      case line.startsWith('Supplementary leverage:'):
        output.supplementaryLeverage = parseFloat(line.replace('Supplementary leverage:', '').trim());
        break;
      case line.startsWith('Abnormal daily return:'):
        output.hasAbnormalDailyReturn = true;
        break;
      case line.startsWith('Last date:'):
        output.lastDate = line.replace('Last date:', '').trim();
        break;
      case line.startsWith('First date:'):
        output.firstDate = line.replace('First date:', '').trim();
        break;
      case line.startsWith('90% confidence interval of annual volatility:'):
        {
          const match = /90% confidence interval of annual volatility: \[(.*), (.*)\]/.exec(line);
          if (!match?.[1] || !match[2]) {
            throw new Error(`Failed to parse annual volatility confidence interval from line: ${line}`);
          }
          output.lowerBoundOfAnnualVolatility = parseFloat(match[1]);
          output.upperBoundOfAnnualVolatility = parseFloat(match[2]);
        }
        break;
      case line.startsWith('80% confidence interval of annual drift minus annual volatility squared halved:'):
        {
          const match = /80% confidence interval of annual drift minus annual volatility squared halved: \[(.*), (.*)\]/.exec(line);
          if (!match?.[1] || !match[2]) {
            throw new Error(`Failed to parse annual drift minus annual volatility squared halved confidence interval from line: ${line}`);
          }
          output.lowerBoundOfAnnualDriftMinusAnnualVolatilitySquaredHalved = parseFloat(match[1]);
          output.upperBoundOfAnnualDriftMinusAnnualVolatilitySquaredHalved = parseFloat(match[2]);
        }
        break;
      case line.startsWith('50% confidence interval of annual drift minus annual volatility squared halved:'):
        {
          const match = /50% confidence interval of annual drift minus annual volatility squared halved: \[(.*), (.*)\]/.exec(line);
          if (!match?.[1] || !match[2]) {
            throw new Error(`Failed to parse annual drift minus annual volatility squared halved confidence interval from line: ${line}`);
          }
          output.sublowerBoundOfAnnualDriftMinusAnnualVolatilitySquaredHalved = parseFloat(match[1]);
          output.subupperBoundOfAnnualDriftMinusAnnualVolatilitySquaredHalved = parseFloat(match[2]);
        }
        break;
      case line.startsWith('80% confidence interval of median of annual return:'):
        {
          const match = /80% confidence interval of median of annual return: \[(.*), (.*)\]/.exec(line);
          if (!match?.[1] || !match[2]) {
            throw new Error(`Failed to parse median of annual return confidence interval from line: ${line}`);
          }
          output.lowerBoundOfMedianOfAnnualReturn = parseFloat(match[1]);
          output.upperBoundOfMedianOfAnnualReturn = parseFloat(match[2]);
        }
        break;
      case line.startsWith('50% confidence interval of median of annual return:'):
        {
          const match = /50% confidence interval of median of annual return: \[(.*), (.*)\]/.exec(line);
          if (!match?.[1] || !match[2]) {
            throw new Error(`Failed to parse median of annual return confidence interval from line: ${line}`);
          }
          output.sublowerBoundOfMedianOfAnnualReturn = parseFloat(match[1]);
          output.subupperBoundOfMedianOfAnnualReturn = parseFloat(match[2]);
        }
        break;
      default:
        throw new Error(`Unexpected line in annual_return.py stdout: ${line}`);
    }
  }
  if (output.leverage === undefined) {
    throw new Error('Missing leverage in annual_return.py stdout');
  }
  if (output.lastDate === undefined) {
    throw new Error('Missing last date in annual_return.py stdout');
  }
  if (output.firstDate === undefined) {
    throw new Error('Missing first date in annual_return.py stdout');
  }
  if (output.lowerBoundOfAnnualVolatility === undefined || output.upperBoundOfAnnualVolatility === undefined) {
    throw new Error('Missing confidence interval of annual volatility in annual_return.py stdout');
  }
  if (output.lowerBoundOfAnnualDriftMinusAnnualVolatilitySquaredHalved === undefined || output.upperBoundOfAnnualDriftMinusAnnualVolatilitySquaredHalved === undefined) {
    throw new Error('Missing confidence interval of annual drift minus annual volatility squared halved in annual_return.py stdout');
  }
  if (output.sublowerBoundOfAnnualDriftMinusAnnualVolatilitySquaredHalved === undefined || output.subupperBoundOfAnnualDriftMinusAnnualVolatilitySquaredHalved === undefined) {
    throw new Error('Missing confidence interval of annual drift minus annual volatility squared halved in annual_return.py stdout');
  }
  if (output.lowerBoundOfMedianOfAnnualReturn === undefined || output.upperBoundOfMedianOfAnnualReturn === undefined) {
    throw new Error('Missing confidence interval of median of annual return in annual_return.py stdout');
  }
  if (output.sublowerBoundOfMedianOfAnnualReturn === undefined || output.subupperBoundOfMedianOfAnnualReturn === undefined) {
    throw new Error('Missing confidence interval of median of annual return in annual_return.py stdout');
  }
  return output as AnnualReturnPyOutput;
}

const annualReturnPyOutputs = new Map<string, AnnualReturnPyOutput>();

await fs.mkdir('public', { recursive: true });

await Promise.all([
  ...configs.map(async config => {
    console.debug('Running annual_return.py for:', config.name);
    const { stdout, stderr } = await util.promisify(execFile)('./annual_return.py', [
      ...['--path-to-json', config.pathToJSON],
      ...(config.pathToSupplementaryJSON ? ['--path-to-supplementary-json', config.pathToSupplementaryJSON] : []),
      ...(config.supplementaryLeverage ? ['--supplementary-leverage', config.supplementaryLeverage.toString()] : []),
      ...['--path-to-plot', `public/${md5Hash(`${config.name} / annual_return.py`)}.png`],
    ])
    if (stderr) {
      throw new Error(stderr);
    }
    await fs.writeFile(`public/${md5Hash(`${config.name} / annual_return.py`)}.txt`, stdout);
    annualReturnPyOutputs.set(config.name, annualReturnPyStdoutParser(stdout));
  }),
])

await fs.unlink('public/index.html');
const indexHTML = createWriteStream('public/index.html', { flags: 'a', encoding: 'utf8' });
indexHTML.write(`
<html>
  <head>
    <title>ETF Analysis</title>
  </head>
  <body>
    <h2>Annual Return Analysis</h2>
    <table border="1">
      <thead>
        <tr>
          <th scope="col">Name</th>
          <th scope="col">Normality</th>
          <th scope="col">Last Date</th>
          <th scope="col">First Date</th>
          <th scope="col">90% CI of Annual Volatility</th>
          <th scope="col">Test w/ α = 10%</th>
          <th scope="col">Test w/ α = 25%</th>
          <th scope="col">Stdout</th>
          <th scope="col">Plot</th>
        </tr>
      </thead>
      <tbody>
`);
for (const config of configs) {
  const output = annualReturnPyOutputs.get(config.name);
  if (!output) {
    throw new Error(`Missing annual_return.py output for ${config.name}`);
  }
  indexHTML.write(`
        <tr>
          <th scope="row">${config.name}</th>
          <td>${output.hasAbnormalDailyReturn ? '&#10060;' : '&#9989;'}</td>
          <td>${output.lastDate}</td>
          <td>${output.firstDate}</td>
          <td>[${output.lowerBoundOfAnnualVolatility.toFixed(2)}, ${output.upperBoundOfAnnualVolatility.toFixed(2)}]</td>
          <td>${output.lowerBoundOfMedianOfAnnualReturn > 1 ? '&#9989;' : '&#10060;'}</td>
          <td>${output.sublowerBoundOfMedianOfAnnualReturn > 1 ? '&#9989;' : '&#10060;'}</td>
          <td><a href="${md5Hash(`${config.name} / annual_return.py`)}.txt">Stdout</a></td>
          <td><a href="${md5Hash(`${config.name} / annual_return.py`)}.png">Plot</a></td>
        </tr>
  `);
}
indexHTML.write(`
      </tbody>
    </table>
  </body>
</html>
`);
indexHTML.end();
