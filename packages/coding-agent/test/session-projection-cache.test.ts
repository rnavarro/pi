import { describe, expect, it } from "vitest";
import { buildSessionProjection, SessionManager, type SessionMessageEntry } from "../src/core/session-manager.ts";

function userMessage(text: string) {
	return {
		role: "user" as const,
		content: text,
		timestamp: 1,
	} as const;
}

function assistantMessage(text: string) {
	return {
		role: "assistant" as const,
		content: [{ type: "text" as const, text }],
		api: "anthropic-messages",
		provider: "anthropic",
		model: "claude-test",
		usage: {
			input: 1,
			output: 1,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 2,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "stop" as const,
		timestamp: 1,
	};
}

/** A manager with `turns` user/assistant pairs already appended. */
function sessionWithTurns(turns: number): SessionManager {
	const manager = SessionManager.inMemory();
	for (let i = 0; i < turns; i++) {
		manager.appendMessage(userMessage(`question ${i}`));
		manager.appendMessage(assistantMessage(`answer ${i}`));
	}
	return manager;
}

describe("session projection cache", () => {
	it("returns the cached projection when nothing has changed", () => {
		// The footer asks for the projection on every render, and the session sits
		// still between appends. Rebuilding it there walks the whole ancestry.
		const manager = sessionWithTurns(3);
		const first = manager.buildSessionProjection();
		const second = manager.buildSessionProjection();

		expect(second).toBe(first);
	});

	it("rebuilds after an append and includes the new message", () => {
		const manager = sessionWithTurns(2);
		const before = manager.buildSessionProjection();
		manager.appendMessage(userMessage("a new question"));
		const after = manager.buildSessionProjection();

		expect(after).not.toBe(before);
		expect(after.messages.length).toBe(before.messages.length + 1);
		expect(JSON.stringify(after.messages)).toContain("a new question");
	});

	it("rebuilds when the leaf moves to an earlier entry", () => {
		const manager = sessionWithTurns(2);
		const entries = manager.getEntries();
		const firstUser = entries.find((entry) => entry.type === "message" && entry.message.role === "user");
		expect(firstUser).toBeDefined();

		const full = manager.buildSessionProjection();
		manager.branch(firstUser!.id);
		const branched = manager.buildSessionProjection();

		expect(branched).not.toBe(full);
		expect(branched.messages.length).toBeLessThan(full.messages.length);
	});

	it("matches a freshly computed projection for the same entries", () => {
		// The cache must not be able to disagree with the pure function it wraps.
		const manager = sessionWithTurns(4);
		const cached = manager.buildSessionProjection();
		const recomputed = buildSessionProjection(manager.getEntries(), manager.getLeafId(), undefined);

		expect(JSON.stringify(cached.messages)).toBe(JSON.stringify(recomputed.messages));
		expect(cached.thinkingLevel).toBe(recomputed.thinkingLevel);
		expect(cached.model).toEqual(recomputed.model);
	});

	it("does not copy every entry to build a projection", () => {
		// getEntries() filters the whole file entry list on each call; a projection
		// that goes through it is O(session) even when the context is small.
		const manager = sessionWithTurns(50);
		let copies = 0;
		const original = manager.getEntries.bind(manager);
		manager.getEntries = () => {
			copies++;
			return original();
		};

		manager.buildSessionProjection();
		manager.buildSessionProjection();

		expect(copies).toBe(0);
	});

	it("keeps the projection linear in ancestry, not quadratic, as the session grows", () => {
		// Two projections of a 2,000-entry session: the memo makes the second free,
		// and the first walks the ancestry once rather than twice.
		const manager = sessionWithTurns(1000);
		const entries = manager.getEntries() as SessionMessageEntry[];
		expect(entries.length).toBe(2000);

		const started = performance.now();
		manager.buildSessionProjection();
		const first = performance.now() - started;
		const secondStarted = performance.now();
		manager.buildSessionProjection();
		const second = performance.now() - secondStarted;

		expect(second).toBeLessThan(Math.max(first / 4, 0.5));
	});
});
