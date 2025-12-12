import asyncio
import random
import re
import time
from collections import deque
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from astrbot.api import logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star, register
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.message.components import At, Plain, Reply
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.star.filter.command import CommandFilter
from astrbot.core.star.filter.command_group import CommandGroupFilter
from astrbot.core.star.star_handler import star_handlers_registry

from .interest import Interest
from .sentiment import Sentiment
from .similarity import Similarity

# 内置指令文本
BUILT_CMDS = [
    "llm",
    "t2i",
    "tts",
    "sid",
    "op",
    "wl",
    "dashboard_update",
    "alter_cmd",
    "provider",
    "model",
    "plugin",
    "plugin ls",
    "new",
    "switch",
    "rename",
    "del",
    "reset",
    "history",
    "persona",
    "tool ls",
    "key",
    "websearch",
]


class MemberState(BaseModel):
    uid: str
    silence_until: float = 0.0  # 沉默到何时
    last_wake: float = 0.0  # 最后唤醒bot的时间
    last_wake_reason: str = ""  # 最后唤醒bot的原因
    last_reply: float = 0.0  # 最后回复的时间
    pend: deque = Field(default_factory=lambda: deque(maxlen=4))  # 事件缓存
    lock: asyncio.Lock = Field(default_factory=asyncio.Lock)
    model_config = ConfigDict(arbitrary_types_allowed=True)


class GroupState(BaseModel):
    gid: str
    members: dict[str, MemberState] = Field(default_factory=dict)
    shutup_until: float = 0.0  # 闭嘴到何时
    bot_msgs: deque = Field(
        default_factory=lambda: deque(maxlen=5)
    )  # Bot消息缓存，共5条


class StateManager:
    """内存状态管理"""

    _groups: dict[str, GroupState] = {}

    @classmethod
    def get_group(cls, gid: str) -> GroupState:
        if gid not in cls._groups:
            cls._groups[gid] = GroupState(gid=gid)
        return cls._groups[gid]


@register("astrbot_plugin_wakepro", "Zhalslar", "...", "...")
class WakeProPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.conf = config
        self.sent = Sentiment()
        self.sim = Similarity()
        self.commands = self._get_all_commands()
        self.wake_prefix = self.context.get_config().get("wake_prefix")
        interest_words_str: list[str] = self.conf["interest_words_str"]
        interest_words_list: list[list[str]] = [
            words_str.split() for words_str in interest_words_str
        ]
        self.interest = Interest(interest_words_list)

    def _get_all_commands(self) -> list[str]:
        """遍历所有注册的处理器获取所有命令"""
        commands = []
        for handler in star_handlers_registry:
            for fl in handler.event_filters:
                if isinstance(fl, CommandFilter):
                    commands.append(fl.command_name)
                    break
                elif isinstance(fl, CommandGroupFilter):
                    commands.append(fl.group_name)
                    break
        logger.debug(f"插件的指令列表：{commands}")
        return commands

        
    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=99)
    async def on_group_msg(self, event: AstrMessageEvent, *args, **kwargs):
        """主入口（兼容可能传入的额外参数）"""
         # 如果 framework 传入了额外参数，记录它们以便排查
        if args or kwargs:
            try:
                logger.debug(f"[wakepro] on_group_msg received extra args={args} kwargs={kwargs}")
            except Exception logger.debug("[wakepro] on_group_msg received extra args (repr failed)")
        try:
            chain = event.get_messages()
            bid: str = event.get_self_id()
            gid: str = event.get_group_id()
            uid: str = event.get_sender_id()
            msg: str = event.message_str
            g: GroupState = StateManager.get_group(gid)

            logger.debug(
                f"[wakepro] 收到群消息 gid={gid} uid={uid} bid={bid} msg={repr(msg)}"
            )

            # 只处理文本
            if not msg:
                logger.debug("[wakepro] 非文本消息，跳过")
                return

            # 群聊黑白名单 / 用户黑名单
            if uid == bid:
                logger.debug("[wakepro] 发送者为bot自身，跳过")
                return
            if self.conf["group_whitelist"] and gid not in self.conf["group_whitelist"]:
                logger.debug(f"[wakepro] 群组{gid}未在白名单中, 忽略此次唤醒")
                return
            if gid in self.conf["group_blacklist"] and not event.is_admin():
                logger.debug(f"[wakepro] 群组{gid}已处于黑名单中, 忽略此次唤醒")
                event.stop_event()
                return
            if uid in self.conf.get("user_blacklist", []):
                logger.debug(f"[wakepro] 用户{uid}已处于黑名单中, 忽略此次唤醒")
                event.stop_event()
                return

            # 更新成员状态
            if uid not in g.members:
                g.members[uid] = MemberState(uid=uid)

            member = g.members[uid]
            now = time.time()

            # 唤醒CD检查
            if now - member.last_wake < self.conf["wake_cd"]:
                logger.debug(f"[wakepro] {uid} 处于唤醒CD中, 忽略此次唤醒")
                event.stop_event()
                return

            # 唤醒违禁词检查
            if self.conf["wake_forbidden_words"]:
                for word in self.conf["wake_forbidden_words"]:
                    if not event.is_admin() and word in event.message_str:
                        logger.debug(
                            f"[wakepro] {uid} 消息中含有唤醒屏蔽词 '{word}', 忽略此次唤醒"
                        )
                        event.stop_event()
                        return

            # 屏蔽内置指令
            if self.conf["block_builtin"]:
                if not event.is_admin() and event.message_str in BUILT_CMDS:
                    logger.debug(f"[wakepro] {uid} 触发内置指令, 忽略此次唤醒")
                    event.stop_event()
                    return

            # 闭嘴检查
            if g.shutup_until > now:
                logger.debug(f"[wakepro] Bot处于闭嘴中, 忽略此次{uid}的唤醒")
                event.stop_event()
                return

            # 沉默检查（辱骂/人机）
            if not event.is_admin() and member.silence_until > now:
                logger.debug(f"[wakepro] {uid} 处于沉默中, 忽略此次唤醒")
                event.stop_event()
                return

            # 复读屏蔽
            if self.conf["block_reread"]:
                cleaned_msg = re.sub(r"[^\w\u4e00-\u9fff]", "", msg).lower()
                cleaned_bot_msgs = [
                    re.sub(r"[^\w\u4e00-\u9fff]", "", bmsg).lower() for bmsg in g.bot_msgs
                ]
                if cleaned_msg in cleaned_bot_msgs:
                    logger.debug(
                        f"[wakepro] {uid} 发送了与Bot缓存消息相同的内容（忽略符号和空格），触发复读屏蔽"
                    )
                    event.stop_event()
                    return

            # 判断是否是指令
            cmd = msg.split(" ", 1)[0]
            is_cmd = cmd in self.commands or cmd in BUILT_CMDS
            logger.debug(f"[wakepro] is_cmd={is_cmd} cmd={cmd}")

            # 消息缓存与合并（为减少竞态：在这个阶段只 set_extra，真正的 append 在后面并在 lock 中执行）
            if not is_cmd:
                # 先设置 extras（这些操作与 event 自身相关，非对 shared pend 的并发写）
                event.set_extra("orig_message", event.message_str)
                event.set_extra("timestamp", now)
                logger.debug(
                    f"[wakepro] 已设置 event extras orig_message/timestamp (timestamp={now}) for uid={uid}"
                )

                # 防御性地尝试合并：获取 member.pend 的最后一项的 timestamp（在 lock 内安全读取）
                async with member.lock:
                    try:
                        if member.pend:
                            prev = member.pend[-1]
                            prev_ts = None
                            try:
                                prev_ts = prev.get_extra("timestamp")
                            except Exception as e:
                                logger.debug(
                                    f"[wakepro] 获取 pending[-1] 的 timestamp 时发生异常: {e!r}. pending[-1]={repr(prev)}"
                                )
                                prev_ts = None

                            if prev_ts is None:
                                logger.debug(
                                    "[wakepro] pending[-1] 没有 timestamp，跳过合并。pending 内容摘要: "
                                    + ", ".join(
                                        [
                                            f"(type={type(x).__name__}, ts={repr(_safe_get_extra(x, 'timestamp'))}, msg={truncate_repr(_safe_get_extra(x, 'orig_message'))})"
                                            for x in list(member.pend)
                                        ]
                                    )
                                )
                            else:
                                # 尝试将 prev_ts 转为 float 然后比较
                                try:
                                    prev_ts_val = float(prev_ts)
                                except (TypeError, ValueError):
                                    logger.debug(
                                        f"[wakepro] pending[-1] timestamp 无法转 float ({repr(prev_ts)}), 跳过合并"
                                    )
                                    prev_ts_val = None

                                if (
                                    prev_ts_val is not None
                                    and (now - prev_ts_val) < self.conf["pend_cd"]
                                ):
                                    # 合并所有缓存消息到当前 event
                                    msgs: list[str] = [
                                        _safe_get_extra(e, "orig_message") or "" for e in member.pend
                                    ]
                                    # stop 旧的 events，保留它们在 pend 以便后续逻辑（或清理）
                                    for e in member.pend:
                                        try:
                                            e.stop_event()
                                        except Exception as e_ex:
                                            logger.debug(
                                                f"[wakepro] 停止 pending event 时出错: {e_ex!r}"
                                            )
                                    event.message_str = "。".join(msgs + [event.message_str])
                                    logger.debug(
                                        f"[wakepro] 已合并{len(member.pend)}条缓存消息：{event.message_str}"
                                    )
                                else:
                                    logger.debug(
                                        f"[wakepro] 不满足合并时间条件 now - prev_ts={now - (prev_ts_val or 0)} >= pend_cd={self.conf['pend_cd']}"
                                    )
                    except Exception:  # 防护：任何意外都不要让合并逻辑中断整体流程
                        logger.exception("[wakepro] 在合并 pending 消息时发生异常（已捕获）")

            # 各类唤醒条件
            wake = event.is_at_or_wake_command
            reason = ""

            # 前缀唤醒
            if isinstance(self.wake_prefix, list) and self.wake_prefix:
                full_msg = next((seg.text for seg in chain if isinstance(seg, Plain)), "")
                for prefix in self.wake_prefix:
                    if not full_msg.startswith(prefix):
                        continue

                    # 屏蔽前缀指令
                    if (
                        self.conf["block_prefix_cmd"]
                        and not event.is_admin()
                        and is_cmd
                    ):
                        logger.debug(
                            f"[wakepro] {uid} 触发前缀指令, 忽略此次唤醒 (prefix={prefix})"
                        )
                        event.stop_event()
                        return

                    # 屏蔽前缀 LLM（即非指令）
                    if (
                        self.conf["block_prefix_llm"]
                        and not event.is_admin()
                        and not is_cmd
                    ):
                        logger.debug(
                            f"[wakepro] {uid} 触发前缀LLM, 忽略此次唤醒 (prefix={prefix})"
                        )
                        event.stop_event()
                        return

                    # 通过所有过滤，执行唤醒
                    wake = True
                    reason = "prefix"
                    logger.debug(f"[wakepro] {uid} 触发前缀唤醒：{prefix}")
                    break

            # At唤醒 / Reply唤醒
            if not wake:
                for seg in chain:
                    if isinstance(seg, At) and str(seg.qq) == bid:
                        wake = True
                        reason = "at"
                        logger.debug(f"[wakepro] {uid} 触发At唤醒")
                        break
                    elif isinstance(seg, Reply) and str(seg.sender_id) == bid:
                        wake = True
                        reason = "reply"
                        logger.debug(f"[wakepro] {uid} 触发引用回复唤醒")
                        break

            # 提及唤醒
            if not wake and self.conf["mention_wake"]:
                names = [n for n in self.conf["mention_wake"] if n]
                for n in names:
                    if n and n in msg:
                        wake = True
                        reason = "mention"
                        logger.debug(f"[wakepro] {uid} 触发提及唤醒：{n}")
                        break

            # 唤醒延长（如果已经处于唤醒状态且在 wake_extend 秒内，每个用户单独延长唤醒时间）
            if (
                not wake
                and self.conf["wake_extend"]
                and member.last_wake_reason in ["at", "reply", "mention"]
                and (now - member.last_reply) <= int(self.conf["wake_extend"] or 0)
            ):
                wake = True
                reason = "prolong"
                logger.debug(
                    f"[wakepro] {uid} 唤醒延长, 时间为上次llm回复后的第{now - member.last_reply}秒"
                )

            # 话题相关性唤醒
            if not wake and self.conf["relevant_wake"] and g.bot_msgs:
                try:
                    simi = self.sim.similarity(
                        group_id=gid, user_msg=msg, bot_msgs=list(g.bot_msgs)
                    )
                    logger.debug(f"[wakepro] 话题相关度:{simi}")
                    if simi > self.conf["relevant_wake"]:
                        wake = True
                        reason = "similar"
                        logger.debug(
                            f"[wakepro] {uid} 触发话题相关性唤醒, 相关度：{simi}"
                        )
                except Exception:
                    logger.exception("[wakepro] 计算话题相似度时出错（已捕获）")

            # 答疑唤醒
            if (
                not wake
                and self.conf["ask_wake"]
                and (ask_th := self.sent.ask(msg)) > self.conf["ask_wake"]
            ):
                wake = True
                reason = "ask"
                logger.debug(f"[wakepro] {uid} 触发答疑唤醒, 疑问值：{ask_th}")

            # 无聊唤醒
            if (
                not wake
                and self.conf["bored_wake"]
                and (bored_th := self.sent.bored(msg)) > self.conf["bored_wake"]
            ):
                wake = True
                reason = "bored"
                logger.debug(f"[wakepro] {uid} 触发无聊唤醒, 无聊值：{bored_th}")

            # 兴趣唤醒
            if not wake and self.conf["interest_wake"]:
                try:
                    interest_th = self.interest.calc_interest(msg)
                    logger.debug(f"[wakepro] 兴趣值：{interest_th}")
                    if interest_th > self.conf["interest_wake"]:
                        wake = True
                        reason = "interest"
                        logger.debug(
                            f"[wakepro] {uid} 触发兴趣唤醒, 兴趣值：{interest_th}"
                        )
                except Exception:
                    logger.exception("[wakepro] 计算兴趣值时出错（已捕获）")

            # 概率唤醒
            if (
                not wake
                and self.conf["prob_wake"]
                and random.random() < self.conf["prob_wake"]
            ):
                wake = True
                reason = "prob"
                logger.debug(f"[wakepro] {uid} 触发概率唤醒")

            # 触发唤醒
            if wake:
                # 记录唤醒标志与时间（先设置）
                event.is_at_or_wake_command = True
                member.last_wake = now
                member.last_wake_reason = reason
                logger.debug(f"[wakepro] wake=True reason={reason} for uid={uid}")

                # 缓存消息：确保 append 在 lock 下进行，避免竞态
                if cmd not in self.commands:
                    try:
                        async with member.lock:
                            member.pend.append(event)
                            logger.debug(
                                f"[wakepro] 已添加 event 到缓存（在锁内 append）：pend_len={len(member.pend)} uid={uid}"
                            )
                    except Exception:
                        logger.exception("[wakepro] 在向 member.pend append 时发生异常（已捕获）")
            else:
                logger.debug(f"[wakepro] wake=False，未触发唤醒 gid={gid} uid={uid}")

            # 闭嘴机制(对当前群聊闭嘴)
            if self.conf["shutup"]:
                shut_th = self.sent.shut(msg)
                if shut_th > self.conf["shutup"]:
                    silence_sec = shut_th * self.conf["silence_multiple"]
                    g.shutup_until = now + silence_sec
                    reason = f"闭嘴沉默{silence_sec}秒"
                    logger.debug(f"[wakepro] 群({gid}){reason}：{msg}")
                    event.stop_event()
                    return

            # 辱骂沉默机制(对单个用户沉默)
            if self.conf["insult"]:
                insult_th = self.sent.insult(msg)
                if insult_th > self.conf["insult"]:
                    silence_sec = insult_th * self.conf["silence_multiple"]
                    member.silence_until = now + silence_sec
                    reason = "insult"
                    logger.info(f"[wakepro] 群({gid})用户({uid}){reason}：{msg}")
                    # event.stop_event() 本轮对话不沉默，方便回怼
                    return

            # AI沉默机制(对单个用户沉默)
            if self.conf["ai"]:
                ai_th = self.sent.is_ai(msg)
                if ai_th > self.conf["ai"]:
                    silence_sec = ai_th * self.conf["silence_multiple"]
                    member.silence_until = now + silence_sec
                    reason = "silence"
                    logger.info(f"[wakepro] 群({gid})用户({uid}){reason}：{msg}")
                    event.stop_event()
                    return

        except Exception:
            logger.exception("[wakepro] on_group_msg 未捕获异常（已捕获并记录）")

    @filter.on_decorating_result(priority=20)
    async def on_message(self, event: AstrMessageEvent):
        """发送消息前，缓存bot消息，清空用户消息缓存"""
        gid: str = event.get_group_id()
        uid: str = event.get_sender_id()
        result = event.get_result()
        if not gid or not uid or not result:
            logger.debug("[wakepro] on_message: gid/uid/result 为空，跳过")
            return
        g: GroupState = StateManager.get_group(gid)
        # 缓存bot消息
        try:
            plain = result.get_plain_text()
        except Exception:
            plain = repr(result)
            logger.debug("[wakepro] 获取 result.get_plain_text() 时异常，使用 repr(result)")
        g.bot_msgs.append(plain)
        logger.debug(f"[wakepro] 已缓存 bot msg (plain={truncate_repr(plain)}). bot_msgs_len={len(g.bot_msgs)}")

        member = g.members.get(uid)
        if not member:
            logger.debug(f"[wakepro] on_message: 找不到 member uid={uid}, 跳过清空 pend")
            return
        # 记录回复时间
        member.last_reply = time.time()
        # 清空用户消息缓存（在 lock 下清空）
        try:
            async with member.lock:
                pend_len = len(member.pend)
                member.pend.clear()
                logger.debug(f"[wakepro] 在 on_message 中清空 member.pend，之前长度={pend_len} uid={uid}")
        except Exception:
            logger.exception("[wakepro] 在 on_message 清空 member.pend 时发生异常（已捕获）")


# --- 辅助函数（局部私有） -------------------------------------------------
def _safe_get_extra(ev: Any, key: str) -> Any:
    """安全获取 event.extra 中的 key；若不可用则返回 None（并不抛异常）"""
    try:
        return ev.get_extra(key)
    except Exception:
        return None


def truncate_repr(s: Any, length: int = 120) -> str:
    try:
        r = repr(s)
        if len(r) <= length:
            return r
        return r[:length] + "...(truncated)"
    except Exception:
        return "(repr_error)"
