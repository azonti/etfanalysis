import { configs, md5Hash } from './publisher-common.js';
import os from 'node:os';
import path from 'node:path';
import fs from 'node:fs/promises';
import { execFile } from 'node:child_process';
import util from 'node:util';

if (!process.argv[2]) {
  throw new Error('Please provide a config name as a command-line argument.');
}
const selectedConfig = configs.find(config => config.name === process.argv[2]);
if (!selectedConfig) {
  throw new Error(`Config with name "${process.argv[2]}" not found.`);
}

await Promise.all([
  (async () => {
    console.debug('Running annual_return_v2.py for:', selectedConfig.name);
    const mplConfigDir = await fs.mkdtemp(path.join(os.tmpdir(), `matplotlib-${md5Hash(`${selectedConfig.name} / annual_return_v2.py`)}-`));
    const { stdout, stderr } = await util.promisify(execFile)('./annual_return_v2.py', [
      ...['--path-to-json', selectedConfig.pathToJSON],
      ...(selectedConfig.pathToSupplementaryJSON ? ['--path-to-supplementary-json', selectedConfig.pathToSupplementaryJSON] : []),
      ...(selectedConfig.supplementaryLeverage ? ['--supplementary-leverage', selectedConfig.supplementaryLeverage.toString()] : []),
      ...['--path-to-plot', `public/${md5Hash(`${selectedConfig.name} / annual_return_v2.py`)}.png`],
    ], {
      env: {
        ...process.env,
        MPLCONFIGDIR: mplConfigDir,
      },
    });
    if (stderr.trim().split('\n').some(line =>
      line.trim() &&
      !line.trim().endsWith('it/s]') &&
      !line.trim().endsWith('s/it]') &&
      line.trim() !== 'Matplotlib is building the font cache; this may take a moment.'
    )) {
      throw new Error(stderr);
    }
    await fs.writeFile(`public/${md5Hash(`${selectedConfig.name} / annual_return_v2.py`)}.txt`, stdout);
  })(),
  (async () => {
    console.debug('Running annual_return.py for:', selectedConfig.name);
    const mplConfigDir = await fs.mkdtemp(path.join(os.tmpdir(), `matplotlib-${md5Hash(`${selectedConfig.name} / annual_return.py`)}-`));
    const { stdout, stderr } = await util.promisify(execFile)('./annual_return.py', [
      ...['--path-to-json', selectedConfig.pathToJSON],
      ...(selectedConfig.pathToSupplementaryJSON ? ['--path-to-supplementary-json', selectedConfig.pathToSupplementaryJSON] : []),
      ...(selectedConfig.supplementaryLeverage ? ['--supplementary-leverage', selectedConfig.supplementaryLeverage.toString()] : []),
      ...['--path-to-plot', `public/${md5Hash(`${selectedConfig.name} / annual_return.py`)}.png`],
    ], {
      env: {
        ...process.env,
        MPLCONFIGDIR: mplConfigDir,
      },
    });
    if (stderr.trim().split('\n').some(line =>
      line.trim() &&
      line.trim() !== 'Matplotlib is building the font cache; this may take a moment.'
    )) {
      throw new Error(stderr);
    }
    await fs.writeFile(`public/${md5Hash(`${selectedConfig.name} / annual_return.py`)}.txt`, stdout);
  })(),
])
