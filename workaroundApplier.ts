import fs from 'node:fs/promises';
import { parse as dateParse, isBefore as isDateBefore } from 'date-fns';

interface UnrecognizedStockSplit {
  readonly pathToJSON: string;
  readonly date: string;
  readonly ratio: number;
}

const unrecognizedStockSplits: UnrecognizedStockSplit[] = [
  {
    pathToJSON: './N225_JST_JPY/1579.T_JST_JPY.json',
    date: '2024-07-02',
    ratio: 0.01,
  },
  {
    pathToJSON: './N225_JST_JPY/1579.T_JST_JPY.json',
    date: '2024-07-01',
    ratio: 100,
  },
  {
    pathToJSON: './N225_JST_JPY/1579.T_JST_JPY.json',
    date: '2024-06-28',
    ratio: 100,
  },
  {
    pathToJSON: './N225_JST_JPY/1579.T_JST_JPY.json',
    date: '2021-03-05',
    ratio: 0.5,
  },
  {
    pathToJSON: './N225_JST_JPY/1579.T_JST_JPY.json',
    date: '2021-03-03',
    ratio: 2,
  },
  {
    pathToJSON: './N225_JST_JPY/1579.T_JST_JPY.json',
    date: '2013-05-09',
    ratio: 2,
  },
  {
    pathToJSON: './TOPIX_JST_JPY/1306.T_JST_JPY.json',
    date: '2026-03-30',
    ratio: 10,
  },
  {
    pathToJSON: './TOPIX_JST_JPY/1568.T_JST_JPY.json',
    date: '2024-07-02',
    ratio: 0.01,
  },
  {
    pathToJSON: './TOPIX_JST_JPY/1568.T_JST_JPY.json',
    date: '2024-07-01',
    ratio: 100,
  },
  {
    pathToJSON: './TOPIX_JST_JPY/1568.T_JST_JPY.json',
    date: '2024-06-28',
    ratio: 100,
  },
  {
    pathToJSON: './TOPIX_JST_JPY/1568.T_JST_JPY.json',
    date: '2015-06-18',
    ratio: 2,
  },
  {
    pathToJSON: './TOPIX_JST_JPY/1568.T_JST_JPY.json',
    date: '2012-04-05',
    ratio: 0.5,
  },
];

const unrecognizedStockSplitsByPath = new Map<string, UnrecognizedStockSplit[]>();
for (const unrecognizedStockSplit of unrecognizedStockSplits) {
  if (!unrecognizedStockSplitsByPath.has(unrecognizedStockSplit.pathToJSON)) {
    unrecognizedStockSplitsByPath.set(unrecognizedStockSplit.pathToJSON, []);
  }
  unrecognizedStockSplitsByPath.get(unrecognizedStockSplit.pathToJSON)!.push(unrecognizedStockSplit); // eslint-disable-line @typescript-eslint/no-non-null-assertion
}

for (const [pathToJSON, unrecognizedStockSplits] of unrecognizedStockSplitsByPath) {
  const data = JSON.parse(await fs.readFile(pathToJSON, 'utf-8')) as string[][];
  for (const entry of data) {
    for (const unrecognizedStockSplit of unrecognizedStockSplits) {
      /* eslint-disable @typescript-eslint/no-non-null-assertion */
      if (isDateBefore(dateParse(entry[0]!, 'MMM dd, yyyy', new Date()), dateParse(unrecognizedStockSplit.date, 'yyyy-MM-dd', new Date()))) {
        entry[1] = String(parseFloat(entry[1]!.replaceAll(',', '')) / unrecognizedStockSplit.ratio);
        entry[2] = String(parseFloat(entry[2]!.replaceAll(',', '')) / unrecognizedStockSplit.ratio);
        entry[3] = String(parseFloat(entry[3]!.replaceAll(',', '')) / unrecognizedStockSplit.ratio);
        entry[4] = String(parseFloat(entry[4]!.replaceAll(',', '')) / unrecognizedStockSplit.ratio);
        entry[5] = String(parseFloat(entry[5]!.replaceAll(',', '')) / unrecognizedStockSplit.ratio);
        entry[6] = String(parseFloat(entry[6]!.replaceAll(',', '')) * unrecognizedStockSplit.ratio);
      }
      /* eslint-enable @typescript-eslint/no-non-null-assertion */
    }
  }
  await fs.writeFile(pathToJSON, JSON.stringify(data));
}
