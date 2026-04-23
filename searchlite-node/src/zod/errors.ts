// Errors raised by the Zod-schema compile path.
//
// Each error carries the field `path` (dot notation, matching Zod's issue
// paths) and a remediation `hint` that points the user at the explicit
// workaround.

export class UnsupportedZodTypeError extends Error {
	readonly path: string;
	readonly zodType: string;
	readonly hint: string;

	constructor(args: { path: string; zodType: string; hint: string }) {
		const path = args.path || "<root>";
		super(
			`searchlite compile: field \`${path}\` — unsupported Zod type ${args.zodType}. ${args.hint}`,
		);
		this.name = "UnsupportedZodTypeError";
		this.path = args.path;
		this.zodType = args.zodType;
		this.hint = args.hint;
	}
}

export class InvalidZodSchemaError extends Error {
	readonly path: string;

	constructor(args: { path: string; message: string }) {
		const path = args.path || "<root>";
		super(`searchlite compile: field \`${path}\` — ${args.message}`);
		this.name = "InvalidZodSchemaError";
		this.path = args.path;
	}
}
