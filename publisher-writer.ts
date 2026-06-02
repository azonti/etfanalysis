import { configs, md5Hash } from './publisher-common.js';
import fs from 'node:fs/promises';
import { createWriteStream } from 'node:fs';

interface AnnualReturnV2PyOutput {
  leverage: number;
  supplementaryLeverage?: number;
  hasAbnormalDailyReturn?: boolean;
  lastDate: string;
  firstDate: string;
  lowerBoundOfAnnualLongRunVolatility: number;
  upperBoundOfAnnualLongRunVolatility: number;
  lowerBoundOfMonteCarloIntervalOfMDD: number;
  upperBoundOfMonteCarloIntervalOfMDD: number;
  sublowerBoundOfMonteCarloIntervalOfMDD: number;
  subupperBoundOfMonteCarloIntervalOfMDD: number;
}

function annualReturnV2PyStdoutParser(stdout: string): AnnualReturnV2PyOutput {
  const lines = stdout.trim().split('\n');
  const output: Partial<AnnualReturnV2PyOutput> = {};
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
      case line.startsWith('Current parameters:'):
        break;
      case line.startsWith('Current loss:'):
        break;
      case line.startsWith('Eigenvalues of Hessian:'):
        break;
      case line.startsWith('90% confidence interval of annual long-run volatility:'):
        {
          const match = /90% confidence interval of annual long-run volatility: \[(.*), (.*)\]/.exec(line);
          if (!match?.[1] || !match[2]) {
            throw new Error(`Failed to parse confidence interval of annual long-run volatility from line: ${line}`);
          }
          output.lowerBoundOfAnnualLongRunVolatility = parseFloat(match[1]);
          output.upperBoundOfAnnualLongRunVolatility = parseFloat(match[2]);
        }
        break;
      case line.startsWith('90% Monte Carlo interval of MDD in 3 years:'):
        {
          const match = /90% Monte Carlo interval of MDD in 3 years: \[(.*), (.*)\]/.exec(line);
          if (!match?.[1] || !match[2]) {
            throw new Error(`Failed to parse Monte Carlo interval of MDD from line: ${line}`);
          }
          output.lowerBoundOfMonteCarloIntervalOfMDD = parseFloat(match[1]);
          output.upperBoundOfMonteCarloIntervalOfMDD = parseFloat(match[2]);
        }
        break;
      case line.startsWith('50% Monte Carlo interval of MDD in 3 years:'):
        {
          const match = /50% Monte Carlo interval of MDD in 3 years: \[(.*), (.*)\]/.exec(line);
          if (!match?.[1] || !match[2]) {
            throw new Error(`Failed to parse Monte Carlo interval of MDD from line: ${line}`);
          }
          output.sublowerBoundOfMonteCarloIntervalOfMDD = parseFloat(match[1]);
          output.subupperBoundOfMonteCarloIntervalOfMDD = parseFloat(match[2]);
        }
        break;
      default:
        throw new Error(`Unexpected line in annual_return_v2.py stdout: ${line}`);
    }
  }
  if (output.leverage === undefined) {
    throw new Error('Missing leverage in annual_return_v2.py stdout');
  }
  if (output.lastDate === undefined) {
    throw new Error('Missing last date in annual_return_v2.py stdout');
  }
  if (output.firstDate === undefined) {
    throw new Error('Missing first date in annual_return_v2.py stdout');
  }
  if (output.lowerBoundOfAnnualLongRunVolatility === undefined || output.upperBoundOfAnnualLongRunVolatility === undefined) {
    throw new Error('Missing confidence interval of annual long-run volatility in annual_return_v2.py stdout');
  }
  if (output.lowerBoundOfMonteCarloIntervalOfMDD === undefined || output.upperBoundOfMonteCarloIntervalOfMDD === undefined) {
    throw new Error('Missing Monte Carlo interval of MDD in annual_return_v2.py stdout');
  }
  if (output.sublowerBoundOfMonteCarloIntervalOfMDD === undefined || output.subupperBoundOfMonteCarloIntervalOfMDD === undefined) {
    throw new Error('Missing Monte Carlo interval of MDD in annual_return_v2.py stdout');
  }
  return output as AnnualReturnV2PyOutput;
}

interface AnnualReturnPyOutput {
  leverage: number;
  supplementaryLeverage?: number;
  hasAbnormalDailyReturn?: boolean;
  lastDate: string;
  firstDate: string;
  lowerBoundOfAnnualVolatility: number;
  upperBoundOfAnnualVolatility: number;
  lowerBoundOfAnnualDriftMinusHalvedSquaredAnnualVolatility: number;
  upperBoundOfAnnualDriftMinusHalvedSquaredAnnualVolatility: number;
  sublowerBoundOfAnnualDriftMinusHalvedSquaredAnnualVolatility: number;
  subupperBoundOfAnnualDriftMinusHalvedSquaredAnnualVolatility: number;
  lowerBoundOfMonteCarloIntervalOfMDD: number;
  upperBoundOfMonteCarloIntervalOfMDD: number;
  sublowerBoundOfMonteCarloIntervalOfMDD: number;
  subupperBoundOfMonteCarloIntervalOfMDD: number;
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
            throw new Error(`Failed to parse confidence interval of annual volatility from line: ${line}`);
          }
          output.lowerBoundOfAnnualVolatility = parseFloat(match[1]);
          output.upperBoundOfAnnualVolatility = parseFloat(match[2]);
        }
        break;
      case line.startsWith('80% confidence interval of annual drift minus halved squared annual volatility:'):
        {
          const match = /80% confidence interval of annual drift minus halved squared annual volatility: \[(.*), (.*)\]/.exec(line);
          if (!match?.[1] || !match[2]) {
            throw new Error(`Failed to parse confidence interval of annual drift minus halved squared annual volatility from line: ${line}`);
          }
          output.lowerBoundOfAnnualDriftMinusHalvedSquaredAnnualVolatility = parseFloat(match[1]);
          output.upperBoundOfAnnualDriftMinusHalvedSquaredAnnualVolatility = parseFloat(match[2]);
        }
        break;
      case line.startsWith('50% confidence interval of annual drift minus halved squared annual volatility:'):
        {
          const match = /50% confidence interval of annual drift minus halved squared annual volatility: \[(.*), (.*)\]/.exec(line);
          if (!match?.[1] || !match[2]) {
            throw new Error(`Failed to parse confidence interval of annual drift minus halved squared annual volatility from line: ${line}`);
          }
          output.sublowerBoundOfAnnualDriftMinusHalvedSquaredAnnualVolatility = parseFloat(match[1]);
          output.subupperBoundOfAnnualDriftMinusHalvedSquaredAnnualVolatility = parseFloat(match[2]);
        }
        break;
      case line.startsWith('90% Monte Carlo interval of MDD in 3 years:'):
        {
          const match = /90% Monte Carlo interval of MDD in 3 years: \[(.*), (.*)\]/.exec(line);
          if (!match?.[1] || !match[2]) {
            throw new Error(`Failed to parse Monte Carlo interval of MDD from line: ${line}`);
          }
          output.lowerBoundOfMonteCarloIntervalOfMDD = parseFloat(match[1]);
          output.upperBoundOfMonteCarloIntervalOfMDD = parseFloat(match[2]);
        }
        break;
      case line.startsWith('50% Monte Carlo interval of MDD in 3 years:'):
        {
          const match = /50% Monte Carlo interval of MDD in 3 years: \[(.*), (.*)\]/.exec(line);
          if (!match?.[1] || !match[2]) {
            throw new Error(`Failed to parse Monte Carlo interval of MDD from line: ${line}`);
          }
          output.sublowerBoundOfMonteCarloIntervalOfMDD = parseFloat(match[1]);
          output.subupperBoundOfMonteCarloIntervalOfMDD = parseFloat(match[2]);
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
  if (output.lowerBoundOfAnnualDriftMinusHalvedSquaredAnnualVolatility === undefined || output.upperBoundOfAnnualDriftMinusHalvedSquaredAnnualVolatility === undefined) {
    throw new Error('Missing confidence interval of annual drift minus halved squared annual volatility in annual_return.py stdout');
  }
  if (output.sublowerBoundOfAnnualDriftMinusHalvedSquaredAnnualVolatility === undefined || output.subupperBoundOfAnnualDriftMinusHalvedSquaredAnnualVolatility === undefined) {
    throw new Error('Missing confidence interval of annual drift minus halved squared annual volatility in annual_return.py stdout');
  }
  if (output.lowerBoundOfMonteCarloIntervalOfMDD === undefined || output.upperBoundOfMonteCarloIntervalOfMDD === undefined) {
    throw new Error('Missing Monte Carlo interval of MDD in annual_return.py stdout');
  }
  if (output.sublowerBoundOfMonteCarloIntervalOfMDD === undefined || output.subupperBoundOfMonteCarloIntervalOfMDD === undefined) {
    throw new Error('Missing Monte Carlo interval of MDD in annual_return.py stdout');
  }
  return output as AnnualReturnPyOutput;
}

const annualReturnV2PyOutputs = new Map<string, AnnualReturnV2PyOutput>();
const annualReturnPyOutputs = new Map<string, AnnualReturnPyOutput>();

await fs.mkdir('public', { recursive: true });

await Promise.all([
  ...configs.map(async config => {
    const stdout = await fs.readFile(`public/${md5Hash(`${config.name} / annual_return_v2.py`)}.txt`, { encoding: 'utf8' });
    annualReturnV2PyOutputs.set(config.name, annualReturnV2PyStdoutParser(stdout));
  }),
  ...configs.map(async config => {
    const stdout = await fs.readFile(`public/${md5Hash(`${config.name} / annual_return.py`)}.txt`, { encoding: 'utf8' });
    annualReturnPyOutputs.set(config.name, annualReturnPyStdoutParser(stdout));
  }),
])

await fs.rm('public/index.html', { force: true });
const indexHTML = createWriteStream('public/index.html', { flags: 'a', encoding: 'utf8' });
indexHTML.write(`
<html>
  <head>
    <title>ETF Analysis</title>
  </head>
  <body>
    <h2>Annual Return Analysis (V2)</h2>
    <table border="1">
      <thead>
        <tr>
          <th scope="col">Name</th>
          <th scope="col">Normality</th>
          <th scope="col">Last Date</th>
          <th scope="col">First Date</th>
          <th scope="col">90% CI of Annual Long-Run Vol</th>
          <th scope="col">90% MCI of MDD in 3 yrs</th>
          <th scope="col">50% MCI of MDD in 3 yrs</th>
          <th scope="col">Stdout</th>
          <th scope="col">Plot</th>
        </tr>
      </thead>
      <tbody>
`);
for (const config of configs) {
  const output = annualReturnV2PyOutputs.get(config.name);
  if (!output) {
    throw new Error(`Missing annual_return_v2.py output for ${config.name}`);
  }
  indexHTML.write(`
        <tr>
          <th scope="row">${config.name}</th>
          <td>${output.hasAbnormalDailyReturn ? '&#10060;' : '&#9989;'}</td>
          <td>${output.lastDate}</td>
          <td>${output.firstDate}</td>
          <td>[${output.lowerBoundOfAnnualLongRunVolatility.toFixed(2)}, ${output.upperBoundOfAnnualLongRunVolatility.toFixed(2)}]</td>
          <td>[${output.lowerBoundOfMonteCarloIntervalOfMDD.toFixed(2)}, ${output.upperBoundOfMonteCarloIntervalOfMDD.toFixed(2)}]</td>
          <td>[${output.sublowerBoundOfMonteCarloIntervalOfMDD.toFixed(2)}, ${output.subupperBoundOfMonteCarloIntervalOfMDD.toFixed(2)}]</td>
          <td><a href="${md5Hash(`${config.name} / annual_return_v2.py`)}.txt">Stdout</a></td>
          <td><a href="${md5Hash(`${config.name} / annual_return_v2.py`)}.png">Plot</a></td>
        </tr>
  `);
}
indexHTML.write(`
      </tbody>
    </table>
    <h2>Annual Return Analysis</h2>
    <table border="1">
      <thead>
        <tr>
          <th scope="col">Name</th>
          <th scope="col">Normality</th>
          <th scope="col">Last Date</th>
          <th scope="col">First Date</th>
          <th scope="col">90% CI of Annual Vol</th>
          <th scope="col">Test w/ α = 10%</th>
          <th scope="col">Test w/ α = 25%</th>
          <th scope="col">90% MCI of MDD in 3 yrs</th>
          <th scope="col">50% MCI of MDD in 3 yrs</th>
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
          <td>${output.lowerBoundOfAnnualDriftMinusHalvedSquaredAnnualVolatility > 0 ? '&#9989;' : '&#10060;'}</td>
          <td>${output.sublowerBoundOfAnnualDriftMinusHalvedSquaredAnnualVolatility > 0 ? '&#9989;' : '&#10060;'}</td>
          <td>[${output.lowerBoundOfMonteCarloIntervalOfMDD.toFixed(2)}, ${output.upperBoundOfMonteCarloIntervalOfMDD.toFixed(2)}]</td>
          <td>[${output.sublowerBoundOfMonteCarloIntervalOfMDD.toFixed(2)}, ${output.subupperBoundOfMonteCarloIntervalOfMDD.toFixed(2)}]</td>
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
