/**
 * Strip control characters and Unicode bidi/invisible/format characters from
 * untrusted recalled content before it is rendered back to the model — these
 * are used for prompt-injection and spoofing. Tab (9), LF (10) and CR (13) are
 * preserved. Implemented as a code-point filter (no literal control chars in
 * source).
 */
export function sanitizeUntrusted(text: string): string {
	let out = "";
	for (const ch of text) {
		const c = ch.codePointAt(0) ?? 0;
		if (c === 9 || c === 10 || c === 13) {
			out += ch;
			continue;
		}
		// C0 controls + DEL + C1 controls.
		if (c <= 0x1f || (c >= 0x7f && c <= 0x9f)) continue;
		// Zero-width, bidi overrides/isolates, word joiner, BOM.
		if (
			(c >= 0x200b && c <= 0x200f) ||
			(c >= 0x202a && c <= 0x202e) ||
			(c >= 0x2060 && c <= 0x2064) ||
			(c >= 0x2066 && c <= 0x2069) ||
			c === 0xfeff
		) {
			continue;
		}
		out += ch;
	}
	return out;
}
