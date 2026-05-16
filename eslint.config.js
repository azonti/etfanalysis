// @ts-check

import eslint from '@eslint/js';
import { defineConfig } from 'eslint/config';
import tseslint from 'typescript-eslint';

export default defineConfig(
  eslint.configs.recommended,
  tseslint.configs.strictTypeChecked,
  tseslint.configs.stylisticTypeChecked,
  {
    languageOptions: {
      ecmaVersion: 'latest',
      parserOptions: {
        projectService: true,
      },
    },
  },
  {
    rules: {
      'no-undef': 'off',
    },
  },
  {
    files: ['**/*.{js,cjs}'],
    extends: [tseslint.configs.disableTypeChecked],
  },
  {
    files: ['**/*.{js,cjs}'],
    rules: {
      'no-undef': 'error',
    },
  },
);
