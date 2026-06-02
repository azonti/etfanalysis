import { configs } from './publisher-common.js';

console.log(`[${configs.map(config => `"${config.name}"`).join(',')}]`);
