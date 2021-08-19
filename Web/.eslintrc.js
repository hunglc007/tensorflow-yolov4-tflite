module.exports = {
	parser: '@typescript-eslint/parser',
	parserOptions: {
		//We want to use the latest ECMAScript standards
		ecmaVersion: 2019,
		//Set the source to ES Modules
		sourceType: 'module',
		//Allow jsx (Or in our case tsx)
		ecmaFeatures: {
			jsx: true,
		},
	},
	plugins: ['@typescript-eslint', 'react', 'jsx-a11y', 'prettier', 'emotion', 'react-hooks'],
	env: {
		browser: true,
		jest: true,
		node: true,
	},
	extends: [
		//An ESlint plugin to maintain A11y standards while developing
		'plugin:jsx-a11y/recommended',
		'plugin:@typescript-eslint/recommended',
		//Prettier comes last to override formatting rules from eslint's recommended
		'prettier',
		'prettier/react',
		'prettier/@typescript-eslint',
	],
	settings: {
		react: {
			version: 'detect',
		},
	},
	rules: {
		'@typescript-eslint/ban-ts-ignore': 'warn',
		'@typescript-eslint/interface-name-prefix': 'off',
		'@typescript-eslint/no-empty-function': 'warn',
		'@typescript-eslint/explicit-function-return-type': [
			'error',
			{
				allowExpressions: true,
				allowTypedFunctionExpressions: true,
			},
		],
		'react/jsx-uses-vars': 1,
		'react/jsx-uses-react': 1,
		'react-hooks/rules-of-hooks': 'error',
		'react-hooks/exhaustive-deps': 'warn',
	},
	globals: {
		page: true,
		browser: true,
		context: true,
		jestPuppeteer: true,
	},
};
